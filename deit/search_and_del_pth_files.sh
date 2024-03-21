

# (active_finetune) wenshuai@node19:~/projects/ActiveFT_xu/deit$ find ./ -size +100M -name "*.pth" |wc -l
# 126
# (active_finetune) wenshuai@node19:~/projects/ActiveFT_xu/deit$ find ./ -name "*.pth" |wc -l
# 127
# reserve the pretrained file: ./ckpt/dino_deitsmall16_pretrain.pth (86M)


# cd deit
find ./ -size +100M -name "*.pth";
find ./ -size +100M -name "*.pth" -exec rm {} \;


