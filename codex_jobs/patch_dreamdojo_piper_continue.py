from pathlib import Path


def insert_before(path: str, marker: str, block: str, sentinel: str) -> None:
    p = Path(path)
    text = p.read_text()
    if sentinel in text:
        return
    if marker not in text:
        raise SystemExit(f"missing marker in {path}: {marker!r}")
    p.write_text(text.replace(marker, block + marker, 1))


def append_once(path: str, block: str, sentinel: str) -> None:
    p = Path(path)
    text = p.read_text()
    if sentinel in text:
        return
    p.write_text(text.rstrip() + "\n" + block)


def replace_once(path: str, old: str, new: str, sentinel: str) -> None:
    p = Path(path)
    text = p.read_text()
    if sentinel in text:
        return
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


append_once(
    "cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_warmup.py",
    """
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_piper_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_PIPER),
)
""",
    "cosmos_predict2p5_2B_action_piper_warmup_no_s3",
)

self_forcing = "cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_self_forcing.py"
insert_before(
    self_forcing,
    "cs = ConfigStore.instance()",
    """
ACTION_GR00T_PIPER_SELF_FORCING = make_experiment(
    name="piper",
    data="gr00t_customized_piper_long",
    overrides=dict(
        job=dict(project="dreamdojo", group="interactive_self_forcing"),
        checkpoint=dict(load_path="", strict_resume=False),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model",
                    credentials="",
                ),
            ),
        ),
    ),
)

""",
    "ACTION_GR00T_PIPER_SELF_FORCING",
)
append_once(
    self_forcing,
    """
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_piper_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_PIPER_SELF_FORCING),
)
""",
    "cosmos_predict2p5_2B_action_piper_self_forcing_no_s3",
)

action_exp = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py"
insert_before(
    action_exp,
    "cs = ConfigStore.instance()",
    """
DREAMDOJO_2B_1440_640_PIPER = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release_oss",
            {"override /data_train": "gr00t_customized_piper"},
            {"override /data_val": "gr00t_customized_piper"},
            "_self_",
        ],
        job=dict(group="piper", name="dreamdojo_2b_1440_640_piper", project="dreamdojo"),
        model=dict(config=dict(state_t=1 + 12 // 4, net=dict(action_dim=384))),
        dataloader_train=dict(batch_size=4),
    ),
    flags={"allow_objects": True},
)


""",
    "DREAMDOJO_2B_1440_640_PIPER",
)
old = """    [
        AC_CHUNK_MULTI_VIEW_2B_GR00T_GR1_CUSTOMIZED_13FRAME_FULL_16NODES_OSS,
        *build_debug_runs(AC_CHUNK_MULTI_VIEW_2B_GR00T_GR1_CUSTOMIZED_13FRAME_FULL_16NODES_OSS),
    ],
"""
new = old + """    [
        DREAMDOJO_2B_1440_640_PIPER,
        *build_debug_runs(DREAMDOJO_2B_1440_640_PIPER),
    ],
"""
replace_once(action_exp, old, new, "DREAMDOJO_2B_1440_640_PIPER,\n        *build_debug_runs")

print("patched_continue")
