# train movement prediction model
system("cd ./scripts/movement_model")
system("python train.py")
system("cd ../..")
system("python scripts/movement_predictor.py")
# train counterattack success prediction model
system("python scripts/convert_pyg.py")
system("python graph_train.py")
# countereval
system("python scripts/test_eval.py")
system("python scripts/eval_case_study.py")