import argparse

def define_arguments_prompted(parser):
    parser.add_argument("--model_type", help="Type of the model to evaluate")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default='', required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", default=False, action='store_true', help="Only predict the first step")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    parser.add_argument("--data_dir", default="./data", help="Folder with test_cases_{dh,ds}_{setting}.json (e.g. data/channels/pdf/pymupdf/white)")
    # Parallel runs and the vLLM server (--model_type VLLM)
    parser.add_argument("--num_workers", type=int, default=1, help="Concurrent requests. >1 only for server-backed models (VLLM, GPT, Claude, TogetherAI)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases per file (smoke tests); results go to a separate _limitN folder")
    parser.add_argument("--vllm_base_url", default=None, help="vLLM server URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)")
    parser.add_argument("--max_tokens", type=int, default=512, help="VLLM: max new tokens (512 = same as HFChat)")
    parser.add_argument("--temperature", type=float, default=0.0, help="VLLM: 0 = greedy (same as HFChat)")
    parser.add_argument("--seed", type=int, default=None, help="VLLM: per-request seed (matters only when temperature > 0)")
    parser.add_argument("--stop_at_observation", default=False, action='store_true', help="VLLM: stop generation at '\\nObservation:' (protocol change vs. InjecAgent; report it if used)")
    parser.add_argument("--chat_template_kwargs", default=None, help='VLLM: JSON passed to the chat template, e.g. \'{"enable_thinking": false}\' for Qwen3')
    parser.add_argument("--request_timeout", type=float, default=600, help="VLLM: seconds per request")

    
def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", action='store_true', help="Only predict the first step")
    
def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params