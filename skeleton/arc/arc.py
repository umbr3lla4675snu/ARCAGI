from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from utils import system_prompt, user_message_template1, user_message_template2, user_message_template3 #.
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from torch.utils.data import Dataset, DataLoader  # ✨[NEW]
import os                                        # ✨[NEW]
import json
from pathlib import Path
import numpy as np

"""
Lets go team7, Check the README.md for more information.
This is a template for the ARC project. Have fun!
"""
class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.bfloat16 # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.bfloat16,  # ← 기존: torch.float16 # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def format_prompt(self, datapoint, training_mode=False):
        """
        Args:
            datapoint (dict): contains training data, test input
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt, add_special_tokens=False)
        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n", add_special_tokens=False)
        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += inp_desc
        user += self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)


        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis

        if training_mode:
            # ✅ 학습 시에는 정답 출력을 붙여야 함
            answer_tokens = self.format_grid(datapoint['test'][0]['output'])
            messages += answer_tokens

        return {
            "input_ids": messages,
            "input": input_test_data,
            "train": training_data
        }


    def train(self, train_dataset):
        """
        Train a model with train_dataset.
        Read a project documentation for a description of `examples` and `question`.
        """
        """
        ✨[NEW] Naive training loop using teacher forcing and prompt tokens as labels
        """
        class ARCDataset(Dataset):
            def __init__(self, dataset, tokenizer, format_prompt_fn):
                self.data = []
                self.tokenizer = tokenizer
                for item in dataset:
                    prompt = format_prompt_fn(item, training_mode=True)
                    input_ids = prompt['input_ids']
                    labels = input_ids.copy()
                    self.data.append((input_ids, labels))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                input_ids, labels = self.data[idx]
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }

        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            labels = [item["labels"] for item in batch]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            return {"input_ids": input_ids, "labels": labels}

        dataset = ARCDataset(train_dataset, self.tokenizer, self.format_prompt)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

        self.model.train()
        self.model.to(self.device)

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step}: loss = {loss.item():.4f}")

        save_dir = "artifacts/checkpoint-final"
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)        

    def predict(self, examples, questions_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        prompt = self.format_prompt(datapoint)
        input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)

        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=150,
        )

        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.numel()

        output = output[N_prompt:].tolist()
        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
            y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]

        try:
            grid = np.array(self.parse_grid(output))
            grid = grid[:x, :y]
            
        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.model.load_adapter("artifacts/checkpoint-final")
        self.model.eval()

def load_training_data(base_dir, N=300):
    filenames = [f for f in os.listdir(base_dir) if f.endswith(".json")]
    dataset = []
    rng = np.random.default_rng(1337)

    for file in filenames[:N]:  # limit for now
        with open(os.path.join(base_dir, file)) as fp:
            task_data = json.load(fp)
        if len(task_data) < 4:
            continue
        for _ in range(1):  # create 1 training instance per task
            idx = rng.choice(len(task_data), size=4, replace=False)
            train_data = [task_data[i] for i in idx[:3]]
            test_example = task_data[idx[3]]  # ✅ test_input + output 모두 포함

            datapoint = {
                "train": train_data,
                "test": [ 
                    { 
                        "input": test_example["input"], 
                        "output": test_example["output"]  # ✅ 정답 포함!
                    }
                ]
            }
            dataset.append(datapoint)
    return dataset

if __name__ == "__main__":
    #solver = ARCSolver()
    #loaded_data = load_json_dataset("/workspace/dataset")  # 실제 데이터 경로
    #solver.train(loaded_data)
    token = os.environ.get("HF_TOKEN", None)
    solver = ARCSolver(token=token)

    train_data = load_training_data("/workspace/dataset", N=300)
    solver.train(train_data)