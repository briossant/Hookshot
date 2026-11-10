"""The mesh fallback orders its own buffers with targeted barriers, not four full ones a draw.

A full barrier waits for every stage of all the work recorded before it. fallback_record had four a draw (before the shared
records are reset, after the reset, between the task and mesh dispatches, before the draw), and a dense Hyrule Field view has
about 116 fallback draws: 464 full barriers a frame, the GPU idling between many small dispatches on an RX 5500M. Each now
covers only the stages that touched the fallback's buffers last and those that touch them next; what the fallback reads of the
game's memory is ordered by the game's barriers, and the pass after it starts with pass_start_barrier.
RX_VK_FALLBACK_FULL_BARRIERS=1 gives the full barriers back (an A/B).

python -I -B tests/test_mesh_fallback_barriers.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAW = ROOT / "gpu" / "vk" / "vk_draw.cpp"


def body(src, head):
    start = src.index(head)
    return src[start:src.index("\n}\n", start)]


def main():
    draw = DRAW.read_text(encoding="utf-8")
    helper = body(draw, "void fallback_barrier(")
    if '"RX_VK_FALLBACK_FULL_BARRIERS"' not in helper or "return full_barrier(cmd);" not in helper:
        raise SystemExit("RX_VK_FALLBACK_FULL_BARRIERS=1 does not give the full barriers back")
    if "vkCmdPipelineBarrier2(cmd, &di);" not in helper:
        raise SystemExit("the targeted barrier is not recorded")
    record = body(draw, "bool fallback_record(")
    if "full_barrier(cmd)" in record:
        raise SystemExit("fallback_record still records a full barrier")
    calls = record.split("fallback_barrier(cmd, ")[1:]
    if len(calls) != 4:
        raise SystemExit("fallback_record has %d targeted barriers, not 4" % len(calls))
    reset, header, task_mesh, draw_after = (c[:c.index(");")] for c in calls)
    # (before the reset: the last fallback draw's indirect read, vertex stage and dispatches, before the fills and dispatches)
    for stage in ("DRAW_INDIRECT_BIT", "VERTEX_SHADER_BIT", "COMPUTE_SHADER_BIT"):
        if stage not in reset.split(",")[0]:
            raise SystemExit("the reset does not wait for the last fallback draw's %s" % stage)
    if "ALL_TRANSFER_BIT" not in reset or "TRANSFER_WRITE_BIT" not in reset:
        raise SystemExit("the reset's fills are not ordered after the last fallback draw")
    # (after the fills: the dispatches' atomics and the mesh dispatch's indirect read of the header)
    if not header.startswith("VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT") or "INDIRECT_COMMAND_READ_BIT" not in header:
        raise SystemExit("the header's fills are not visible to the dispatches and the indirect read")
    # (task -> mesh: the map, payloads and the dispatch count)
    if not task_mesh.startswith("VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT") or "INDIRECT_COMMAND_READ_BIT" not in task_mesh:
        raise SystemExit("the task's writes are not visible to the mesh dispatch")
    # (mesh -> the draw and everything after it)
    if not draw_after.startswith("VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT") or "ALL_COMMANDS_BIT" not in draw_after:
        raise SystemExit("the mesh dispatch's writes are not visible to the draw and the stages after it")
    # (the indirect arguments converted before the pass order themselves: transfer -> indirect read)
    conv = body(draw, "void mesh_indirect_convert(") if "void mesh_indirect_convert(" in draw else body(draw, "bool mesh_indirect_convert(")
    if "VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT" not in conv:
        raise SystemExit("the converted indirect arguments relied on the fallback's full barriers")
    print("mesh fallback: targeted barriers, full ones only with RX_VK_FALLBACK_FULL_BARRIERS=1")


if __name__ == "__main__":
    main()
