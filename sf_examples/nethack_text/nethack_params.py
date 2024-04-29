import ast


def add_extra_params_nethack_env(parser):
    p = parser
    p.add_argument("--character", type=str, default="@")
    p.add_argument("--max_episode_steps", type=int, default=100000000)
    p.add_argument("--penalty_step", type=float, default=0.0)
    p.add_argument("--penalty_time", type=float, default=0.0)
    p.add_argument("--no_progress_timeout", type=int, default=100)
    p.add_argument("--fn_penalty_step", type=str, default="constant")
    p.add_argument("--savedir", type=str, default=None)
    p.add_argument("--save_ttyrec_every", type=int, default=0)
    p.add_argument("--gameloaddir", type=ast.literal_eval, default=None)
    p.add_argument("--state_counter", type=str, default=None)
    p.add_argument("--add_stats_to_info", type=ast.literal_eval, default=True)
    p.add_argument("--unroll_length", type=int, default=1)
    p.add_argument("--nethack_anchor_every", type=int, default=4)
    p.add_argument("--max_ctx_tokens", type=int, default=8000)
    p.add_argument("--debug_render", type=ast.literal_eval, default=False)


def add_extra_params_general(parser):
    p = parser
    p.add_argument("--exp_tags", type=str, default="local")
    p.add_argument("--exp_point", type=str, default="point-A")
    p.add_argument("--group", type=str, default="group2")
    p.add_argument("--model", type=str)
    p.add_argument("--model_path", type=str)
    p.add_argument("--tokenizer_path", type=str)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_num_batched_tokens", type=int, default=2048)


def nethack_override_defaults(_env, parser):
    # TODO:
    parser.set_defaults()
