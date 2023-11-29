import optuna_class
import pathlib
import constants


main_path = pathlib.Path(__file__).parents[0]

net = constants.Models.LSTM.value
src_dataset = constants.Subsets.FD003.value
trg_dataset = constants.Subsets.FD002.value
window_type = constants.WindowType.Variable.value
deci_mat_method = constants.GM.gtw.value
gm_method = constants.GM.vgm.value
initiator = constants.GM.zeros.value

runner = optuna_class.OptunaOptim(main_path=main_path, 
                                  src_dataset_num=src_dataset, 
                                  trg_dataset_num=trg_dataset,
                                  net=net, 
                                  window_type=window_type,
                                  deci_mat_method=deci_mat_method,
                                  gm_method=gm_method,
                                  initiator=initiator)

runner.run_objective(n_trials=200, start_up_trials=50)
runner.create_summary() 
