import numpy as np

def rates(list_train):
    '''
    to find the rates at which each agent is moving on the road
    :param list_train: list of all train datasets with adjacent matrices
    :return: list of rates for all the datasets
    '''

    all_rates_list = []

    count = 1
    for list1 in list_train:

        print(count)

        diags_list = []

        for item in list1:
            adj = item['adj_matrix']

            d_vals = []
            for item in adj:
                row_sum = sum(item)
                d_vals.append(row_sum)
            diag_array = np.diag(d_vals)
            laplacian = diag_array - adj
            diags_list.append(np.asarray(d_vals))


        all_rates_arr = np.empty([270,1])
        prev_ = diags_list[0]
        for items in range(1, len(diags_list)):
            next_ = diags_list[items]

            rate = next_ - prev_

            all_rates_arr = np.column_stack((all_rates_arr, rate))

        all_rates_arr = np.delete(all_rates_arr , 0, 1)
        all_rates_list.append(all_rates_arr)
        count +=1

    return all_rates_list

