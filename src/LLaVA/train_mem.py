from llava.train.train_with_patch_two_position import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
