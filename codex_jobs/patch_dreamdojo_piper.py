from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


path = "cosmos_predict2/_src/predict2/interactive/configs/data.py"
old = '''dataset_gr00t_pretrain_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/pretrain_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

'''
new = old + '''dataset_gr00t_piper_warmup = L(ActionDatasetSFWarmup)(
    data_path="datasets/piper_warmup_regenerated_4step",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

'''
replace_once(path, old, new)

old = '''gr00t_customized_pretrain_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=pretrain_path,
    dataset_mixing_weights=pretrain_mixing_weights,
    data_split="train",
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt"
)

'''
new = old + '''piper_path, piper_mixing_weights = get_data_path("piper")
gr00t_customized_piper_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=piper_path,
    dataset_mixing_weights=piper_mixing_weights,
    data_split="train",
    height=1440,
    width=640,
    video_key="video.cam_vertical",
    fps=10,
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)
gr00t_customized_piper_dataset_long = L(MultiVideoActionDataset)(
    num_frames=49,
    dataset_path=piper_path,
    dataset_mixing_weights=piper_mixing_weights,
    data_split="train",
    height=1440,
    width=640,
    video_key="video.cam_vertical",
    fps=10,
    cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
)

'''
replace_once(path, old, new)

old = '''        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_pretrain_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_pretrain_warmup),
        )
'''
new = old + '''        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_piper_warmup",
            node=L(make_dataloader)(dataset=dataset_gr00t_piper_warmup),
        )
'''
replace_once(path, old, new)

old = '''        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_pretrain_long",
            node=L(make_dataloader)(dataset=gr00t_customized_pretrain_dataset_long, num_workers=0, pin_memory=False),
        )
'''
new = old + '''        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_piper",
            node=L(make_dataloader)(dataset=gr00t_customized_piper_dataset, num_workers=0, pin_memory=False),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="gr00t_customized_piper_long",
            node=L(make_dataloader)(dataset=gr00t_customized_piper_dataset_long, num_workers=0, pin_memory=False),
        )
'''
replace_once(path, old, new)

path = "cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_warmup.py"
old = '''ACTION_GR00T_WARMUP_PRETRAIN = make_experiment(
    name="pretrain",
    data="gr00t_pretrain_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="checkpoints/warmup/pretrain/iter_000140000",
        ),
    ),
)

'''
new = old + '''ACTION_GR00T_WARMUP_PIPER = make_experiment(
    name="piper",
    data="gr00t_piper_warmup",
    overrides=dict(
        job=dict(project="dreamdojo", group="interactive_warmup"),
        checkpoint=dict(load_path="", strict_resume=False),
        model=dict(config=dict(fps=10)),
    ),
)

'''
replace_once(path, old, new)

old = '''cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_pretrain_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_PRETRAIN),
)
'''
new = old + '''cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_piper_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_PIPER),
)
'''
replace_once(path, old, new)

path = "cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_self_forcing.py"
old = '''ACTION_GR00T_PRETRAIN_SELF_FORCING = make_experiment(
    name="pretrain",
    data="gr00t_customized_pretrain_long",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="checkpoints/self_forcing/pretrain/iter_000010000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="checkpoints/warmup/pretrain/iter_000140000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

'''
new = old + '''ACTION_GR00T_PIPER_SELF_FORCING = make_experiment(
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

'''
replace_once(path, old, new)

old = '''cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_pretrain_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_PRETRAIN_SELF_FORCING),
)
'''
new = old + '''cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_piper_self_forcing_no_s3",
    node=build_no_s3_run(ACTION_GR00T_PIPER_SELF_FORCING),
)
'''
replace_once(path, old, new)

path = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py"
old = '''AC_CHUNK_MULTI_VIEW_2B_GR00T_GR1_CUSTOMIZED_13FRAME_FULL_16NODES_OSS = LazyDict(
    dict(
        defaults=[
            "/experiment/2b_bridge_action_conditioned_oss",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "gr00t_customized_gr1"},
            {"override /data_val": "gr00t_customized_gr1"},
            "_self_",
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes_release_oss",
            project="cosmos_predict2_action_conditioned",
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=384,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            dataset=dict(num_frames=13, data_split="full"),
        ),
        optimizer=dict(
            lr=16e-5,
            weight_decay=0.1,
        ),
    ),
    flags={"allow_objects": True},
)


'''
new = old + '''DREAMDOJO_2B_1440_640_PIPER = LazyDict(
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


'''
replace_once(path, old, new)

old = '''    [
        AC_CHUNK_MULTI_VIEW_2B_GR00T_GR1_CUSTOMIZED_13FRAME_FULL_16NODES_OSS,
        *build_debug_runs(AC_CHUNK_MULTI_VIEW_2B_GR00T_GR1_CUSTOMIZED_13FRAME_FULL_16NODES_OSS),
    ],
'''
new = old + '''    [
        DREAMDOJO_2B_1440_640_PIPER,
        *build_debug_runs(DREAMDOJO_2B_1440_640_PIPER),
    ],
'''
replace_once(path, old, new)

print("patched")
