import json
import logging
from typing import List, Dict, Any, Union, Generator

import datasets
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.renderers import get_renderer, TrainOnWhat, Renderer

# Define Features matching the expected schema
FEATURES = datasets.Features(
    {
        "messages": datasets.Sequence(
            {
                "role": datasets.Value("string"),
                "content": datasets.Sequence(
                    {
                        "type": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "thinking": datasets.Value("string"),
                    }
                ),
                "tool_calls": datasets.Sequence(
                    {
                        "function": {
                            "name": datasets.Value("string"),
                            "arguments": datasets.Value("string"),
                        },
                        "id": datasets.Value("string"),
                        "type": datasets.Value("string"),
                    }
                ),
                "tool_call_id": datasets.Value("string"),
                "name": datasets.Value("string"),
            }
        )
    }
)

def _convert_gemini_part_to_tool_call(part: Dict[str, Any]) -> Dict[str, Any]:
    fc = part["functionCall"]
    name = fc["name"]
    args = fc["args"]
    # Ensure args is a generic JSON string
    args_str = json.dumps(args) if isinstance(args, (dict, list)) else str(args)

    return {
        "function": {
            "name": name,
            "arguments": args_str,
        },
        "id": None,
        "type": "function",
    }

def _convert_row(row: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = []

    for msg in row:
        role = msg["role"]
        parts = msg.get("parts", [])

        # Determine new role
        new_role = "user"
        if role == "model":
            new_role = "assistant"
        elif role == "user":
            # Check if it has functionResponse
            if any("functionResponse" in p for p in parts):
                new_role = "tool"

        content_parts = []
        tool_calls = []
        tool_name: Union[str, None] = None

        for p in parts:
            if "text" in p:
                text = p["text"]
                if p.get("thought", False):
                    content_parts.append({"type": "thinking", "thinking": text})
                else:
                    content_parts.append({"type": "text", "text": text})
            elif "functionCall" in p:
                tool_calls.append(_convert_gemini_part_to_tool_call(p))
            elif "functionResponse" in p:
                resp = p["functionResponse"]
                # For tool response, content needs to be the response
                content_parts.append({"type": "text", "text": json.dumps(resp["response"])})
                tool_name = resp["name"]

        # Construct Message matching renderers.base.Message
        out_msg = {
            "role": new_role,
            "content": content_parts,  # Always list[ContentPart] for consistency
            "tool_calls": tool_calls,  # datasets Sequence expects list
            "tool_call_id": None,
            "name": tool_name if tool_name else None,
        }
        messages.append(out_msg)

    return messages

def _get_sub_trajectories(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Generates sub-trajectories ending with an assistant message."""
    sub_trajectories = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            sub_trajectories.append(messages[:i+1])
    return sub_trajectories

def gemini_history_generator(filepaths: Union[str, List[str]], generate_sub_traj: bool = False, **kwargs) -> Generator[Dict[str, Any], None, None]:
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = _convert_row(row)
                
                if generate_sub_traj:
                    for sub_traj in _get_sub_trajectories(messages):
                        yield {"messages": sub_traj}
                else:
                    yield {"messages": messages}

from tinker_cookbook.renderers import ToolCall

def _transform_messages_to_objects(batch):
    # batch['messages'] is a list of list of dicts
    transformed_messages = []
    
    for conversation in batch['messages']:
        new_conversation = []
        for msg in conversation:
            # We want to preserve the dict but convert tool_calls field
            # msg is a dict here (from dataset)
            new_msg = msg.copy() 
            
            # Handle tool_calls if present and not None
            if new_msg.get('tool_calls'):
                # Convert each dict in the list to ToolCall object
                new_msg['tool_calls'] = [
                    ToolCall(**tc) if isinstance(tc, dict) else tc 
                    for tc in new_msg['tool_calls']
                ]
            
            new_conversation.append(new_msg)
        transformed_messages.append(new_conversation)
        
    return {"messages": transformed_messages}

def load_gemini_history(tokenizer: Tokenizer, renderer: Renderer, data_files: Union[str, List[str]] = "gemini-3.0-preview-history.jsonl", generate_sub_traj: bool = False, batch_size: int = 4) -> SupervisedDataset:
    """
    Load the Gemini history dataset using datasets.Dataset.from_generator.
    Returns a SupervisedDatasetFromHFDataset wrapper for compatibility with tinker's training loop.
    Use .hf_dataset to access the underlying HuggingFace dataset.
    """
    ds = datasets.Dataset.from_generator(
        gemini_history_generator,
        gen_kwargs={
            "filepaths": data_files, 
            "generate_sub_traj": generate_sub_traj,
            "cache_buster": "v3_force_refresh" 
        },
        # features=FEATURES, # Rely on inference to avoid structure mismatch issues
    )
    ds.set_transform(_transform_messages_to_objects)
    
    # Wrap in SupervisedDatasetFromHFDataset for compatibility with train.py

    
    def map_fn(row: dict) -> Any: # Returns tinker.Datum
        # row is a dict with "messages" key
        # Because SupervisedDatasetFromHFDataset calls to_list() which bypasses transforms,
        # we might get raw dicts instead of ToolCall objects. We need to handle that.
        from tinker_cookbook.renderers import ToolCall
        
        messages = row["messages"]
        for msg in messages:
            if msg.get("tool_calls"):
                # Check if elements are dicts inside the list
                new_tool_calls = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                         new_tool_calls.append(ToolCall(**tc))
                    else:
                         new_tool_calls.append(tc)
                msg["tool_calls"] = new_tool_calls
        
        return conversation_to_datum(
            messages,
            renderer,
            max_length=4096, # Default max length
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE if generate_sub_traj else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

    return SupervisedDatasetFromHFDataset(
        ds,
        batch_size=batch_size, # Default batch size
        map_fn=map_fn
    )



def test_sub_trajectory_logic():
    print("Testing sub-trajectory logic...")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good"},
    ]
    subs = _get_sub_trajectories(messages)
    assert len(subs) == 2, f"Expected 2 sub-trajectories, got {len(subs)}"
    
    # Check first sub-trajectory
    assert len(subs[0]) == 2
    assert subs[0][-1]["role"] == "assistant"
    assert subs[0][0]["content"] == "Hello"
    assert subs[0][1]["content"] == "Hi"
    
    # Check second sub-trajectory
    assert len(subs[1]) == 4
    assert subs[1][-1]["role"] == "assistant"
    assert subs[1][2]["content"] == "How are you?"
    assert subs[1][3]["content"] == "Good"
    print("Sub-trajectory logic passed!")

if __name__ == "__main__":
    try:
        test_sub_trajectory_logic()
        print("Testing dataset loading with from_generator...")
        tokenizer = get_tokenizer("Qwen/Qwen3-8B")
        renderer = get_renderer("qwen3", tokenizer)
        # renderer = get_renderer("qwen3_disable_thinking", tokenizer)
        ds_wrapper = load_gemini_history(tokenizer, renderer, "k8s-dataset.jsonl", generate_sub_traj=True)
        ds = ds_wrapper.hf_dataset
        print(f"Successfully loaded {len(ds)} conversations.")
        if len(ds) > 0:
            for row_idx, row_data in enumerate(ds):
                messages = row_data.get('messages')
                for msg_idx, msg in enumerate(messages):
                    if msg_idx == len(messages) -1:
                        print(f"{row_idx} {msg_idx} {msg}")
                print(f"\n----------------------\n")
                if row_idx == 3:
                    break
            # first_conv = ds[0]
            # print(f"First conversation has {len(first_conv['messages'])} messages.")
            # print("First message role:", first_conv['messages'][0]['role'])
            
            # # Verify tool calls
            # for conv in ds:
            #     for msg in conv['messages']:
            #         if msg['tool_calls']:
            #             print("Found tool call in a message:", msg['tool_calls'])
            #             break
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading dataset: {e}")

def load_k8s_prompts(filepath: str = "k8s-dataset.jsonl") -> List[str]:
    """
    Load K8s prompts from a JSONL file.
    Propagates the prompt from the first user message in each conversation.
    """
    prompts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                
                # Check if row is a list (some formats) or dict
                # Based on previous file content, the row is actually a list of messages directly? 
                # Wait, the `head` output showed `[{"parts":...}, ...]`
                # So the top level object is a LIST of messages.
                
                messages = row if isinstance(row, list) else row.get("messages", [])
                
                # If row is a dict and has "candidates" it might be raw Gemini API output, 
                # but based on `head`, it looks like a list of message objects.
                # Let's handle the specific format seen in `head`: List[Dict]
                
                first_user_msg = None
                for msg in messages:
                    if msg.get("role") == "user":
                        # Extract text from parts
                        parts = msg.get("parts", [])
                        texts = []
                        for p in parts:
                            if "text" in p:
                                texts.append(p["text"])
                        
                        if texts:
                            first_user_msg = "".join(texts)
                            break
                
                if first_user_msg:
                    prompts.append(first_user_msg)
                    
    except Exception as e:
        logging.warning(f"Could not load k8s prompts from {filepath}: {e}")
        return []

    return prompts
