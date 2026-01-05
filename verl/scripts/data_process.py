import json
import pandas as pd
import os

input_file = "/path/to/your/data/input_file.jsonl"  
output_dir = "/path/to/your/output/directory"  


def process_to_custom_format():
    final_data = []
    
    print(f"Reading from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                messages = item.get("messages", [])
                prompt_msgs = [
                    msg for msg in messages 
                    if msg['role'] in ['system', 'user']
                ]
                assistant_msgs = [
                    msg for msg in messages 
                    if msg['role'] == 'assistant'
                ]
                
                if not prompt_msgs or not assistant_msgs:
                    print(f"Skipping line {i}: Missing prompt or response.")
                    continue
                user_content_text = ""
                for msg in reversed(prompt_msgs):
                    if msg['role'] == 'user':
                        user_content_text = msg['content']
                        break
                
                assistant_content_text = assistant_msgs[0]['content']
                row = {
                    "prompt": prompt_msgs,
                    "data_source": "unknown",  
                    "ability": "rule",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": assistant_content_text
                    },
                    "extra_info": {
                        "id": f"example_{i}", 
                        "en_question": user_content_text, 
                        "en_answer": assistant_content_text 
                    }
                }
                
                final_data.append(row)
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue


    test_list = final_data[:8]
    train_list = final_data[8:]
    df_train = pd.DataFrame(train_list)
    df_test = pd.DataFrame(test_list)
    
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples:  {len(df_test)}")
    train_path = os.path.join(output_dir, "train_data.parquet")  
    test_path = os.path.join(output_dir, "test_data.parquet")  
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    if len(df_train) > 0:
        print("\n=== Data Structure Check ===")
        print("1. Prompt Example (Check System Prompt):")
        print(df_train.iloc[0]['prompt'])
        print("\n2. Reward Model Field:")
        print(df_train.iloc[0]['reward_model'])
        print("\n3. Extra Info Field:")
        print(df_train.iloc[0]['extra_info'])
    

    print(f"\nSaved to {output_dir}")

if __name__ == "__main__":
    process_to_custom_format()
