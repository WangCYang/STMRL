import pickle

if __name__ == "__main__":
    scenario = "Net4"
    # scenario = "bologna_pasubio"

    ### Simulation: 3_3_Grid
    # neighboring_edges_3_3 = {"J6":["J4","J7"],
    #                          "J4": ["J6", "J5", "J0"],
    #                          "J5": ["J4", "J1"],
    #                          "J7": ["J6", "J0", "J8"],
    #                          "J0": ["J4", "J7", "J3","J1"],
    #                          "J1": ["J5", "J2", "J0"],
    #                          "J8": ["J7", "J3"],
    #                          "J3": ["J0", "J8", "J2"],
    #                          "J2": ["J3", "J1"],
    #                          }
    # fw = open("data/3-3-grid/neighbor_edges.pkl", "wb")
    # pickle.dump(neighboring_edges_3_3, fw)

    ### Simulation: Net4
    # center_junction_ids = ['J11', 'J12', 'J0', 'J2', 'J3', 'J17', 'J6', 'J8', 'J9', 'J14', 'J7', 'J10', 'J15']
    # neighboring_edges_net4 = {"J11": ["J0", "J12"],
    #                           "J12":["J11", "J3"],
    #                          "J0": ["J11", "J2", "J17"],
    #                          "J2": ["J0", "J3",  "J8"],
    #                          "J3": ["J2", "J9"],
    #                          "J17": ["J0", "J6", "J14"],
    #                          "J6": ["J17", "J8", "J7"],
    #                          "J8": ["J2", "J6","J10", "J9"],
    #                          "J9": ["J3", "J8", "J15"],
    #                          "J14": ["J7", "J17"],
    #                          "J7": ["J14", "J10", "J6"],
    #                          "J10": ["J7", "J8", "J15"],
    #                          "J15": ["J10", "J9"],
    #                          }
    # fw = open("data/{}/neighbor_edges.pkl".format(scenario), "wb")
    # # pickle.dump(no_neighboring_edges_pasubio, fw)

    '''
            Variant No_neighbors: 通过控制neighboring edges的配置，实现IDQL
        '''
    # center_junction_ids = ['J11', 'J12', 'J0', 'J2', 'J3', 'J17', 'J6', 'J8', 'J9', 'J14', 'J7', 'J10', 'J15']  # 13 servres
    # no_neighboring_edges_pasubio = no_neighboring_edges_pasubio = {"J11": [],
    #     #                                 "J12":[],
    #     #                                 "J0": [],
    #     #                                  "J2": [],
    #     #                                  "J3": [],
    #     #                                  "J17": [],
    #     #                                  "J6": [],
    #     #                                  "J8": [],
    #     #                                  "J9": [],
    #     #                                  "J14": [],
    #     #                                  "J7": [],
    #     #                                  "J10": [],
    #     #                                  "J15": [],
    #     #                                  }
    # fw = open("data/{}/no_neighbor_edges.pkl".format(scenario), "wb")
    # pickle.dump(no_neighboring_edges_pasubio, fw)

    '''
        Variant Global_neighbors: 通过控制neighboring edges的配置，实现global DQL
    '''
    center_junction_ids = ['J11', 'J12', 'J0', 'J2', 'J3', 'J17', 'J6', 'J8', 'J9', 'J14', 'J7', 'J10', 'J15']  # 13 servres # 18 servres
    global_neighboring_edges_pasubio = {"J11": center_junction_ids,
                                        "J12":center_junction_ids,
                                        "J0": center_junction_ids,
                                         "J2": center_junction_ids,
                                         "J3": center_junction_ids,
                                         "J17": center_junction_ids,
                                         "J6": center_junction_ids,
                                         "J8": center_junction_ids,
                                         "J9": center_junction_ids,
                                         "J14": center_junction_ids,
                                         "J7": center_junction_ids,
                                         "J10": center_junction_ids,
                                         "J15": center_junction_ids,
                                         }
    fw = open("data/{}/global_neighbor_edges.pkl".format(scenario), "wb")
    pickle.dump(global_neighboring_edges_pasubio, fw)

    ### Simulation: bologna_pasubio
    # center_junction_ids = ['4', '7', '12', '19', '18', '0', '9', '27', '23', '2', '29', '33', '32', '1', '15', '40',
    #                        '39', '36']  # 18 servres
    # neighboring_edges_pasubio = {'4':["7","12"],
    #                              '7':["4","19"],
    #                              '12':["4","19","9"],
    #                              '19':["7","12","9","18"],
    #                              '18':["19","0","23"],
    #                              '0':["18","1"],
    #                              '9':["12","19","27"],
    #                              '27':["9","29","23"],
    #                              '23':["18","27","2","32"],
    #                              '2':["0","23","1"],
    #                              '29':["27","33","15"],
    #                              '33':["29","32","40"],
    #                              '32':["23","33","39","1"],
    #                              '1':["2","32","36"],
    #                              '15':["29","40"],
    #                              '40':["15","33","39"],
    #                              '39':["40","36","32"],
    #                              '36':["39","1"]
    #                           }
    # fw = open("data/{}/neighbor_edges.pkl".format(scenario), "wb")
    # pickle.dump(neighboring_edges_pasubio, fw)

    '''
        Variant No_neighbors: 通过控制neighboring edges的配置，实现IDQL
    '''
    # center_junction_ids = ['4', '7', '12', '19', '18', '0', '9', '27', '23', '2', '29', '33', '32', '1', '15', '40',
    #                        '39', '36']  # 18 servres
    # no_neighboring_edges_pasubio = {'4': [],
    #                              '7': [],
    #                              '12': [],
    #                              '19': [],
    #                              '18': [],
    #                              '0': [],
    #                              '9': [],
    #                              '27': [],
    #                              '23': [],
    #                              '2': [],
    #                              '29': [],
    #                              '33': [],
    #                              '32': [],
    #                              '1': [],
    #                              '15': [],
    #                              '40': [],
    #                              '39': [],
    #                              '36': []
    #                              }
    # fw = open("data/{}/no_neighbor_edges.pkl".format(scenario), "wb")
    # pickle.dump(no_neighboring_edges_pasubio, fw)

    '''
        Variant Global_neighbors: 通过控制neighboring edges的配置，实现global DQL
    '''
    # center_junction_ids = ['4', '7', '12', '19', '18', '0', '9', '27', '23', '2', '29', '33', '32', '1', '15', '40',
    #                        '39', '36']  # 18 servres
    # global_neighboring_edges_pasubio = {'4': center_junction_ids,
    #                                 '7': center_junction_ids,
    #                                 '12': center_junction_ids,
    #                                 '19': center_junction_ids,
    #                                 '18': center_junction_ids,
    #                                 '0': center_junction_ids,
    #                                 '9': center_junction_ids,
    #                                 '27': center_junction_ids,
    #                                 '23': center_junction_ids,
    #                                 '2': center_junction_ids,
    #                                 '29': center_junction_ids,
    #                                 '33': center_junction_ids,
    #                                 '32': center_junction_ids,
    #                                 '1': center_junction_ids,
    #                                 '15': center_junction_ids,
    #                                 '40': center_junction_ids,
    #                                 '39': center_junction_ids,
    #                                 '36': center_junction_ids
    #                                 }
    # fw = open("data/{}/global_neighbor_edges.pkl".format(scenario), "wb")
    # pickle.dump(global_neighboring_edges_pasubio, fw)