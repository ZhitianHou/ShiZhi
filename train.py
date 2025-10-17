import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from swift.llm import (
    InferArguments, ModelType, SftArguments,
    infer_main, sft_main, get_default_template_type, get_model_tokenizer, get_template, inference
)
from swift.tuners import Swift
import argparse
from utils.data_process import load_data, save_data
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser('TrainingShiZhi')
    parser.add_argument('--dataset', type=str, default='data/train/CCVG/train.jsonl')
    parser.add_argument('--test_dataset', type=str, default='data/test/CCVG/test.jsonl')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=250)
    parser.add_argument('--model_name', type=str,
                        default='YourModelName')
    parser.add_argument('--model_chinese_name', type=str,
                        default='模型名字')

    # anonymous space
    parser.add_argument('--output_path', type=str,
                        default='./output')
    parser.add_argument('--ckpt_path', type=str,
                        default='./output/model_name/version/checkpoint-xxx')
    parser.add_argument('--model_path', type=str,
                        default='/path/to/your/base/model')
    return parser


def sft():
    sft_args = SftArguments(
        model_type=ModelType.qwen2_7b_instruct,
        model_id_or_path=args.model_path,
        sft_type='lora',
        dataset=[f'{args.dataset}'],
        dataset_test_ratio=args.test_size,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        truncation_strategy='truncation_left',
        learning_rate=args.learning_rate,
        output_dir=args.output_path,
        lora_target_modules=['ALL'],
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        model_name=[args.model_chinese_name, args.model_name],
        model_author=['你的名字', 'Your Name']
    )
    result = sft_main(sft_args)
    best_model_checkpoint = result['best_model_checkpoint']
    print(f'best_model_checkpoint: {best_model_checkpoint}')
    torch.cuda.empty_cache()

    return {
        "best_model_checkpoint": best_model_checkpoint,
    }


def custom_inference(
        inference_file_path: str,
        output_path: str,
):
    data = load_data(file_path=inference_file_path, suffix='jsonl')
    # single sample test
    # data = [{"fact": "阜阳市颍州区人民检察院指控：2013年2月19日20时许，被告人黄某醉酒后驾驶皖KT1703号小型轿车沿阜阳市临泉路由西向东行驶至太平巷路段时，撞到由南向北横过道路的李某某，造成李某某受伤的交通事故。经安徽中天司法鉴定所检测，被告人黄某静脉血中乙醇含量为229.5mg／100ml。案发后，被告人黄某赔偿被害人李某某经济损失41000元，并得到被害人的谅解。", "meta": {"relevant_articles": [234], "accusation": ["故意伤害"], "punish_of_money": 0, "criminals": ["郑某"], "term_of_imprisonment": {"death_penalty": "false", "imprisonment": 5, "life_imprisonment": "false"}}}]

    ckpt_dir = args.ckpt_path
    model_type = ModelType.qwen2_0_5b_instruct
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')  # template_type: qwen

    model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'},
                                           model_id_or_path=args.model_path)

    model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)

    model.generation_config.max_new_tokens = 512

    template = get_template(template_type, tokenizer)

    for sample in tqdm(data):
        fact = sample["fact"][:512]
        response, history = inference(model, template, f"事实描述:\n{fact}\n法院推理:\n")
        print(response)
        sample["reasoning"] = response

    save_data(data, file_path=output_path, suffix='jsonl')


if __name__ == '__main__':
    args = get_parser().parse_args()

    # train
    train_params = sft()

    # inference
    custom_inference(
        inference_file_path=args.test_dataset,
        output_path=args.output_path,
    )
