# python train_search.py --gpu 1 --dataset cora > results/eff_cora_gcn.txt &
# python train_search.py --gpu 2 --dataset cora --basemodel GraphSAGE > results/eff_cora_sage.txt &
python evaluate.py --gpu 2 --dataset citeseer --basemodel GraphSAGE --drop_out 0.25 --hiddim 256 --gnn_learning_rate 0.0245 --arch_learning_rate 0.003 --arch_learning_rate_min 0.009 --gnn_weight_decay 0.0001 --arch_weight_decay 0.01 --aug_M 4 --max_lambda 1 --conf 0.3287 --tem 0.6793 --reg_loss l2
# python train_search.py --gpu 1 --dataset cora --basemodel GraphSAGE
# python train_search.py --gpu 2 --dataset citeseer --basemodel GCN
# python train_search.py --gpu 3 --dataset citeseer --basemodel GraphSAGE
# python train_search.py --gpu 5 --dataset cora --basemodel GCN
# python train_search_twitch.py --gpu 3 --dataset Twitch_PT --basemodel GAT > results/pt_gat.txt &
# python train_search_twitch.py --gpu 4 --dataset Twitch_EN --basemodel GCN > results/en_gcn.txt &
# python train_search_twitch.py --gpu 5 --dataset Twitch_EN --basemodel GraphSAGE > results/en_sage.txt &
# python train_search_twitch.py --gpu 6 --dataset Twitch_EN --basemodel GAT > results/en_gat.txt &