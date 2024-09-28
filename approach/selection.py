from approach.selectSeed import selectSeed
class selection(object):
    def __init__(self):
        #得到选择和未选择种子的集合
        selected_test_prob,no_selected_candidate_prob,selected_class_samples,no_selected_class_samples,\
            selected_test_psedu,no_selected_candidate_psedu,selected_real_label,no_selected_real_label=selectSeed()
        ...

