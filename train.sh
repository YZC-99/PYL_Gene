echo "开始训练only_dna..."
python main.py -c only_dna/only_balance_dna_mlp & python main.py -c only_dna/only_balance_dna_pca64_mlp & python main.py -c only_dna/only_balance_dna_vote5_mlp & python main.py -c only_dna/only_balance_dna_vote5_pca64_mlp & python main.py -c only_dna/only_dna_mlp & python main.py -c only_dna/only_dna_pca64_mlp & python main.py -c only_dna/only_dna_vote5_mlp & python main.py -c only_dna/only_dna_vote5_pca64_mlp &

echo "开始训练only_ppi..."
python main.py -c only_dna/only_balance_ppi64_mlp & python main.py -c only_dna/only_balance_ppi64_vote5_mlp & python main.py -c only_dna/only_ppi64_mlp & python main.py -c only_dna/only_ppi64_vote5_mlp &

echo "开始训练dna_ppi..."
python main.py -c balance_dna_pca64_ppi64_mlp & python main.py -c balance_dna_pca64_ppi64_vote5_mlp & python main.py -c balance_dna_ppi64_mlp & python main.py -c balance_dna_ppi64_vote5_mlp & python main.py -c dna_pca64_ppi64_mlp & python main.py -c dna_pca64_ppi64_vote5_mlp & python main.py -c dna_ppi64_mlp & python main.py -c dna_ppi64_vote5_mlp &
