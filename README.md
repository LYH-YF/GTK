# environment
pip install jieba
# GTK
run model "python train.py"
            optional:
                "--use_kg True"    use knowledge graph
                "--mean True"      mean operation to encoder output
                "--resume True"    resume training from last
                "--encoder" rnn    rnn module as encoder
