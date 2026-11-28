/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/Transforms/Passes.h"
#include "asctile/Dialect/AscendC/Utils/Attributes.h"
#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "CrossCoreSyncUtils.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNC
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

bool isSyncTriggerOp(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return (src == ascendc::TPosition::VECCALC &&
                (dst == ascendc::TPosition::A1 || dst == ascendc::TPosition::B1)) ||
               (src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::VECCALC);
    });
}

Operation* findGroupAncestor(Operation* op)
{
    for (Operation* p = op->getParentOp(); p; p = p->getParentOp())
        if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(p))
            return p;
    return nullptr;
}

Value unwrapViewLayer(Value v)
{
    if (auto subOp = v.getDefiningOp<ascendc::LocalTensorSubIndexOp>())
        return subOp.getTensor();
    if (auto castOp = v.getDefiningOp<ascendc::LocalTensorReinterpretCastOp>())
        return castOp.getIn();
    return Value();
}

// If `v` is a loop-carried value of some enclosing scf.for, resolve it to the
// corresponding init value (the pre-loop buffer). Handles three forms:
//   * a block argument that is an scf.for iter-arg,
//   * a value yielded by scf.yield of an enclosing for,
//   * an OpResult of an enclosing for (the loop's returned value).
// All of them carry the same logical buffer across iterations, so they resolve
// to the init value that lives before the loop. `ctx` is the op from which the
// search for enclosing for-loops starts (see resolveCanonical/resolveToRoot).
// If `v` is a loop-carried value of some enclosing scf.for, resolve it to the
// corresponding init value (the pre-loop buffer). Handles three forms:
//   * a block argument that is an scf.for iter-arg,
//   * a value yielded by scf.yield of an enclosing for,
//   * an OpResult of an enclosing for (the loop's returned value).
// All of them carry the same logical buffer across iterations, so they resolve
// to the init value that lives before the loop. `ctx` is the op from which the
// search for enclosing for-loops starts (see resolveCanonical/resolveToRoot).
// Returns a null Value when `v` is not loop-carried.
Value resolveLoopCarried(Value v, Operation* ctx)
{
    // iter-arg -> init value.
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        if (auto* owner = blockArg.getOwner()) {
            if (auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp())) {
                if (owner == forOp.getBody() && blockArg.getArgNumber() >= 1) {
                    unsigned idx = blockArg.getArgNumber() - 1;
                    if (idx < forOp.getInitArgs().size())
                        return forOp.getInitArgs()[idx];
                }
            }
        }
        return Value();
    }
    // yield operand / for-result -> init value, searching enclosing for-loops.
    for (Operation* p = ctx; p; p = p->getParentOp()) {
        auto forOp = dyn_cast<scf::ForOp>(p);
        if (!forOp)
            continue;
        auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (!yieldOp)
            continue;
        for (unsigned i = 0; i < yieldOp.getOperands().size() && i < forOp.getInitArgs().size(); ++i) {
            if (yieldOp.getOperands()[i] == v)
                return forOp.getInitArgs()[i];
        }
        if (auto res = dyn_cast<OpResult>(v)) {
            if (res.getOwner() == forOp) {
                unsigned k = res.getResultNumber();
                if (k < yieldOp.getOperands().size() && k < forOp.getInitArgs().size())
                    return forOp.getInitArgs()[k];
            }
        }
    }
    return Value();
}

// Resolve `v` to its canonical root buffer by unwrapping view aliases
// (subindex/cast) and loop-carried values (iter-arg/yield/for-result) to a
// fixpoint. The root is the pre-loop buffer that a producer's dst and a
// consumer's operand both alias, so two producers writing the same buffer
// (through views or loop carries) dedup to the same root -> same flagId.
// `ctx` seeds the enclosing-for search for loop-carried resolution (see
// resolveLoopCarried): for a producer it is the trigger op, so a for-result of
// that very forOp is matched rather than skipped.
Value resolveCanonical(Value v, Operation* ctx)
{
    DenseSet<Value> visited;
    bool changed = true;
    while (changed) {
        if (!visited.insert(v).second)
            return v;
        changed = false;
        if (Value unwrapped = unwrapViewLayer(v)) {
            v = unwrapped;
            changed = true;
            continue;
        }
        if (Value carried = resolveLoopCarried(v, ctx)) {
            v = carried;
            changed = true;
        }
    }
    return v;
}

// The defining op of the canonical root of `v`. Used to match a consumer's
// operand against a producer's dst root (see getCopyDstRoot / consumesDst).
Operation* resolveToRoot(Value v)
{
    Operation* ctx = v.getDefiningOp();
    if (!ctx) {
        if (auto blockArg = dyn_cast<BlockArgument>(v))
            ctx = blockArg.getOwner()->getParentOp();
    }
    return resolveCanonical(v, ctx).getDefiningOp();
}

// True if `op` consumes a result of `dstDefiningOp`, directly or through a chain
// of view aliases and loop/if carries. This is needed because a producer's copy
// dst is often aliased (subindex/cast) or forwarded through loop iter-args /
// results or scf.if results before reaching the cross-core consumer.
// Two resolution directions are combined:
//   * Identity (resolveToRoot): a consumer operand that aliases back to the
//     same pre-loop buffer as the producer's dst matches directly.
//   * Data flow (worklist): a consumer reading a for/if-result depends on data
//     written inside the loop/if via the yield, which identity-collapses to the
//     init (for loops) or has no init (for if) and must be traced forward
//     through the yield instead.
// A loop iter-arg may carry data from BOTH the init (written before the loop)
// and the yield (written inside the loop), so both are pushed onto the worklist.
bool consumesDst(Operation* op, Operation* dstDefiningOp)
{
    for (Value operand : op->getOperands()) {
        SmallVector<Value> worklist = {operand};
        DenseSet<Value> visited;
        while (!worklist.empty()) {
            Value cur = worklist.pop_back_val();
            if (!visited.insert(cur).second)
                continue;
            if (resolveToRoot(cur) == dstDefiningOp)
                return true;
            if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
                // iter-arg carries both init (pre-loop) and yield (in-loop) data.
                if (auto* owner = blockArg.getOwner()) {
                    if (auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp())) {
                        if (owner == forOp.getBody() && blockArg.getArgNumber() >= 1) {
                            unsigned idx = blockArg.getArgNumber() - 1;
                            if (idx < forOp.getInitArgs().size())
                                worklist.push_back(forOp.getInitArgs()[idx]);
                            if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
                                if (idx < yieldOp.getOperands().size())
                                    worklist.push_back(yieldOp.getOperands()[idx]);
                            }
                        }
                    }
                }
                continue;
            }
            Operation* defOp = cur.getDefiningOp();
            if (!defOp)
                continue;
            if (auto subOp = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp)) {
                worklist.push_back(subOp.getTensor());
            } else if (auto castOp = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(defOp)) {
                worklist.push_back(castOp.getIn());
            } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
                // for-result k -> yield operand k (data written inside the loop).
                unsigned resIdx = cast<OpResult>(cur).getResultNumber();
                if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
                    if (resIdx < yieldOp.getOperands().size())
                        worklist.push_back(yieldOp.getOperands()[resIdx]);
                }
            } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
                // if-result k -> both then-yield and else-yield operand k (data
                // produced inside either branch). Unlike scf.for there is no
                // init value to canonicalize back to, so the result is traced
                // forward to whichever yield produced it.
                if (auto res = dyn_cast<OpResult>(cur)) {
                    unsigned resIdx = res.getResultNumber();
                    if (auto thenYield = ifOp.thenYield()) {
                        if (resIdx < thenYield.getOperands().size())
                            worklist.push_back(thenYield.getOperands()[resIdx]);
                    }
                    if (auto elseYield = ifOp.elseYield()) {
                        if (resIdx < elseYield.getOperands().size())
                            worklist.push_back(elseYield.getOperands()[resIdx]);
                    }
                }
            }
        }
    }
    return false;
}

// Collect consumers of `dstDefiningOp` that live in the opposite group type
// (AIC vs AIV) and appear after `afterOp` in function order. Group ops and ops
// before `afterOp` are skipped.
SmallVector<Operation*> collectUsers(Operation* dstDefiningOp, bool triggerIsAIV, Operation* afterOp)
{
    SmallVector<Operation*> users;
    bool seen = !afterOp;
    dstDefiningOp->getParentOfType<func::FuncOp>()->walk([&](Operation* op) {
        if (op == afterOp) {
            seen = true;
            return;
        }
        if (!seen || isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op))
            return;
        if (!consumesDst(op, dstDefiningOp))
            return;
        if (Operation* group = findGroupAncestor(op); group && isa<ascendc::IfAICOp>(group) == triggerIsAIV)
            users.push_back(op);
    });
    return users;
}

// The canonical defining op of a trigger op's copy dst; two triggers writing the
// same buffer (aliased through views/loop-carries) resolve to the same root.
Operation* getCopyDstRoot(Operation* op)
{
    Value dst = cast<ascendc::DataCopyOp>(op).getDst();
    return resolveCanonical(dst, op).getDefiningOp();
}

// A pending before/after-loop sync to be hoisted outside its owning loop.
struct HoistedSync {
    int32_t flagId;
    bool isAIV;
    ascendc::Pipe pipe;
    Operation* groupOp;
};

class InsertCrossCoreSync {
    using HoistedSyncMap = DenseMap<std::pair<scf::ForOp, int>, HoistedSync>;
    using IfSyncMap = DenseMap<std::pair<scf::IfOp, int>, HoistedSync>;
    OpBuilder& builder;
    bool dualSync;
    // Counter for new flag ids (wraps at maxTensorId).
    int32_t nextFlagId = 0;
    // Maps a producer's dst root to the flagId already assigned to it, so a
    // second trigger writing the same buffer reuses the flag (ping-pong).
    DenseMap<Operation*, int32_t> dstFlagId;
    // Hoisted syncs keyed by (owning loop, flagId), emitted after all groups.
    HoistedSyncMap beforeLoopSync;
    HoistedSyncMap afterLoopSync;
    // Producer SetFlag hoisted outside an enclosing scf.if, keyed by (if, flagId).
    // A set emitted inside one branch would hang a consumer that always waits, so
    // the set is moved to after the if where it fires unconditionally. Deduped by
    // (if, flagId) so two branches producing the same buffer hoist a single set.
    IfSyncMap afterIfSync;
    // Insert the consumer-side carry sync for `users`: a WaitFlag at the start
    // of the (single) consumer group and a SetFlag at its end. Returns the pipe
    // of the last user in that group (used for the before-loop prime SetFlag).
    ascendc::Pipe processUsers(ArrayRef<Operation*> users, int32_t flagId)
    {
        Operation* first = users.front();
        Operation* group = findGroupAncestor(first);
        Operation* last = first;
        for (auto* user : users) {
            if (findGroupAncestor(user) != group)
                break;
            last = user;
        }
        bool dualFlags = dualSync && isa_and_present<ascendc::IfAICOp>(group);
        Block& body = group->getRegion(0).front();
        builder.setInsertionPointToStart(&body);
        createWaitFlag(builder, first->getLoc(), flagId, ascendc::getOpPipeExt(first), dualFlags);
        builder.setInsertionPoint(body.getTerminator());
        ascendc::Pipe lastPipe = ascendc::getOpPipeExt(last);
        createSetFlag(builder, last->getLoc(), flagId, lastPipe, dualFlags);
        return lastPipe;
    }

    // Wrap a single cross-core flag op in an IfAIC/IfAIV group, used to hoist a
    // SetFlag/WaitFlag outside a loop (which itself is not a group op).
    template <typename FlagOp>
    void insertFlagGroup(Operation* groupOp, bool isAIV, int32_t flagId, ascendc::Pipe pipe)
    {
        Location loc = groupOp->getLoc();
        ascir::ConstantOpBuilder consts(builder);
        Operation* group = isAIV ? builder.create<ascendc::IfAIVOp>(loc, TypeRange{}, ValueRange{}).getOperation() :
                                   builder.create<ascendc::IfAICOp>(loc, TypeRange{}, ValueRange{}).getOperation();
        builder.createBlock(&group->getRegion(0));
        builder.create<FlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
        if (!isAIV && dualSync)
            builder.create<FlagOp>(loc, consts.i32(maxTensorId + flagId), crossCoreMode, pipe);
        builder.create<ascendc::YieldOp>(loc);
    }

    void processGroup(Operation* groupOp)
    {
        bool isAIV = isa<ascendc::IfAIVOp>(groupOp);
        SmallVector<Operation*> syncOps = collectOps(groupOp, isSyncTriggerOp);
        if (syncOps.empty())
            return;
        // The immediately enclosing loop owns the in-loop carry syncs, and the
        // hoisted prime (before) / drain (after) bracket exactly this loop.
        scf::ForOp loopOp = groupOp->getParentOfType<scf::ForOp>();
        for (auto* op : syncOps)
            processTrigger(op, isAIV, loopOp, groupOp);
    }

    // Process one trigger op. Does nothing if it has no consumers and is not a
    // re-trigger of an existing flag. For a new dst root this allocates a new
    // flagId and schedules the prime/drain syncs; for a re-trigger it reuses
    // the existing flagId and emits only the in-loop carry pair.
    void processTrigger(Operation* op, bool isAIV, scf::ForOp loopOp, Operation* groupOp)
    {
        Operation* dstDefiningOp = getCopyDstRoot(op);
        if (!dstDefiningOp)
            return;
        ascendc::Pipe producerPipe = ascendc::getOpPipeExt(op);
        auto it = dstFlagId.find(dstDefiningOp);
        bool isSecondTrigger = it != dstFlagId.end();
        int32_t flagId = isSecondTrigger ? it->second : nextFlagId;

        SmallVector<Operation*> users = collectUsers(dstDefiningOp, isAIV, op);
        if (users.empty() && !isSecondTrigger)
            return;
        // Allocate a new flag id for a new dst root. A skipped first trigger
        // leaves both the map and the counter untouched, so a later trigger on
        // the same dst is still treated as the first one.
        if (!isSecondTrigger) {
            nextFlagId = (nextFlagId + 1) % maxTensorId;
            dstFlagId[dstDefiningOp] = flagId;
        }
        Block& groupBody = groupOp->getRegion(0).front();
        bool producerDual = dualSync && !isAIV;
        // In-loop carry: wait for the previous stage's Set at the producer start
        // (only inside a loop; outside a loop there is no carry to wait for and
        // it would deadlock). For a second trigger this waits on the other
        // producer's release; for a first trigger it waits on the before-loop
        // prime.
        if (loopOp) {
            builder.setInsertionPointToStart(&groupBody);
            createWaitFlag(builder, op->getLoc(), flagId, producerPipe, producerDual);
        }
        // After-loop drain (one per new flagId): wait once after the owning loop.
        if (loopOp && !isSecondTrigger)
            afterLoopSync.try_emplace({loopOp, flagId}, HoistedSync{flagId, isAIV, producerPipe, groupOp});
        // Producer SetFlag + consumer-side carry. Emitted for re-triggers too
        // (the producer still signals its write even if this trigger's own
        // consumers were served by the first trigger's syncs).
        if (!users.empty() || isSecondTrigger) {
            // If the producer group sits inside an scf.if, hoist its SetFlag to
            // after the if: a set inside one branch would hang a consumer that
            // always waits when the other branch is taken. After the if the
            // buffer is ready (one branch yielded it), so an unconditional set is
            // correct. Deduped by (if, flagId) so two branches producing the
            // same buffer hoist a single set (the second trigger reuses the
            // flagId and finds the entry already present).
            if (auto ifOp = groupOp->getParentOfType<scf::IfOp>()) {
                afterIfSync.try_emplace({ifOp, flagId}, HoistedSync{flagId, isAIV, producerPipe, groupOp});
            } else {
                builder.setInsertionPoint(groupBody.getTerminator());
                createSetFlag(builder, op->getLoc(), flagId, producerPipe, producerDual);
            }
            ascendc::Pipe pipe = producerPipe;
            if (!users.empty())
                pipe = processUsers(users, flagId);
            // Before-loop prime (one per new flagId): set once before the
            // owning loop to seed the carry; uses the consumer pipe.
            if (!isSecondTrigger && loopOp)
                beforeLoopSync.try_emplace({loopOp, flagId}, HoistedSync{flagId, !isAIV, pipe, groupOp});
        }
    }

    // Emit the hoisted before/after-loop syncs just outside the loop that owns
    // the carry (`loopOp`): the prime SetFlag fires once before that loop, and
    // the drain WaitFlag fires once after it. Also emit the producer SetFlags
    // hoisted outside enclosing scf.if ops (see afterIfSync).
    void emitHoistedSyncs()
    {
        for (auto& [key, sync] : beforeLoopSync) {
            builder.setInsertionPoint(key.first ? key.first : sync.groupOp);
            insertFlagGroup<ascendc::CrossCoreSetFlagOp>(sync.groupOp, sync.isAIV, sync.flagId, sync.pipe);
        }
        for (auto& [key, sync] : afterIfSync) {
            builder.setInsertionPointAfter(key.first);
            insertFlagGroup<ascendc::CrossCoreSetFlagOp>(sync.groupOp, sync.isAIV, sync.flagId, sync.pipe);
        }
        for (auto& [key, sync] : afterLoopSync) {
            builder.setInsertionPointAfter(key.first ? key.first : sync.groupOp);
            insertFlagGroup<ascendc::CrossCoreWaitFlagOp>(sync.groupOp, sync.isAIV, sync.flagId, sync.pipe);
        }
    }

public:
    InsertCrossCoreSync(OpBuilder& builder, bool dualSync) : builder(builder), dualSync(dualSync) {}
    ~InsertCrossCoreSync() = default;

    void run(func::FuncOp funcOp)
    {
        SmallVector<Operation*> groupOps = collectGroupOps(funcOp);
        if (groupOps.size() < 2)
            return;
        for (auto* groupOp : groupOps)
            processGroup(groupOp);
        emitHoistedSyncs();
        funcOp->setAttr(attr::crossCoreFlagId, builder.getI32IntegerAttr(nextFlagId));
    }
};

class InsertCrossCoreSyncPass : public ascendc::impl::InsertCrossCoreSyncBase<InsertCrossCoreSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        bool dualSync = false;
        if (auto attr = getModule(funcOp)->getAttrOfType<IntegerAttr>(attr::cvRatio))
            dualSync = attr.getValue().getSExtValue() == 2;
        OpBuilder builder(funcOp.getContext());
        InsertCrossCoreSync insertCrossCoreSync(builder, dualSync);
        insertCrossCoreSync.run(funcOp);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncPass()
{
    return std::make_unique<InsertCrossCoreSyncPass>();
}
