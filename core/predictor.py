from typing import Dict, List
import pickle

import numpy as np


def predict_song(neighbors: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """predict the ranks of song ids for each hum given its retrieved song neighbors
    The most importance job is choose the first place song
    The rules are, given one hum query and its retrieved neighbors:
        1. if in top 10, there is no song id that appear 2 times then the rank 
            follow the distance rank
        2. if in top 10, there is a song that appear >= 3 times, it must be ranked first place
        3. if in top 10, there are song ids that appear 2 times and their ranks < 5, it will be ranked first
        4. if in top 10, there are more than one song id that appear >= 2 times, 
            choose the one that has rank sum smaller to be top 1, then the second rank is the next

        the other positions will follow the distance rank given that it is not already in the ranked list.
    """
    # first we only choose the first rank song
    # assume song_ids are all ints
    ranked_ = {}
    for qname, nbs in neighbors.items():
        chosen = []
        # if one song appear more than 3 times in top 5, it must be the one
        # if the nearest song is not ranked first by this rule, it must be ranked second
        ids, counts = np.unique(nbs[:5], return_counts = True)
        max_count = np.max(counts)
        if max_count >=3:
            idx = list(counts).index(max_count)
            chosen.append(ids[idx])

            if nbs[0] != chosen[0]:
                chosen.append(nbs[0])
            ranked_[qname] = chosen
            continue
        
        # if in top 5 there are *2* song_ids that both appear 2 times, then the one 
        # that on top 1 and appear 2 times will be the first, the one on top 2 
        # or larger and appear 2 times  will be the second
        ids, counts = np.unique(nbs[:5], return_counts = True)
        max_count = np.max(counts)
        if len(ids) == 3 and max_count == 2:
            nearest_song = nbs[0]
            idx_of_nearest_song = list(ids).index(nearest_song)
            count_of_nearest_song = counts[idx_of_nearest_song]
            if count_of_nearest_song == 2:
                chosen.append(nearest_song)
                for i, c in enumerate(counts):
                    if c == 2 and ids[i] not in chosen:
                        chosen.append(ids[i])
            
            ranked_[qname] = chosen
            continue

        # if in top 5, there is *one* song_id that appear 2 times and one of that is
        # top 1, then it must be the one
        # if that song_id appear 2 times but not the nearest, then it still ranked 
        # top 1 but the second ranked is the nearest
        ids, counts = np.unique(nbs[:5], return_counts = True)
        if len(ids) == 4:
            nearest_song_id = nbs[0]
            idx_of_nearest_song = list(ids).index(nearest_song_id)
            if counts[idx_of_nearest_song] == 2:
                chosen.append(nearest_song_id)
                ranked_[qname] = chosen
                continue
            elif counts[idx_of_nearest_song] == 1:
                idx = list(counts).index(2)
                song_id  = ids[idx]
                chosen.append(song_id)
                chosen.append(nearest_song_id)


        # if top 10 are 10 different songs, the just take those
        ids, counts = np.unique(nbs[:10], return_counts = True)
        if len(ids) == 10:
            chosen  = nbs[:10]
            ranked_[qname] = list(chosen)
            continue

        # if in top 5, there are 5 different song ids, and there is one or more
        # song_ids that also appear on top 10 and on top 5, then it will be the
        # first rank, the second rank is the one that nearest(if the previous is 
        # not the nearest)
        ids, counts = np.unique(nbs[:5], return_counts = True)
        if len(ids) == 5:   # also means max_count == 1
            new_ids, new_counts = np.unique(nbs[5:10], return_counts = True)
            for id in nbs[:5]:
                if int(id) in new_ids:
                    chosen.append(id)
            if len(chosen) == 0:
                chosen = list(nbs[:10])
                ranked_[qname] = chosen
                continue
            if chosen[0] != nbs[0]:
                chosen.append(nbs[0])
            ranked_[qname] = chosen
            continue


        if len(chosen) == 0:
            ranked_[qname] = list(nbs[:10])

    # now add the remaining neighbors to the rank list, follow the distance rank
    for qname, ranks in ranked_.items():
        if len(ranks) == 0:
            print('ranks=0')
        j = 0
        while len(ranks) < 10 and j < len(neighbors[qname]):
            if neighbors[qname][j] not in ranks:
                ranks.append(neighbors[qname][j])
            j+=1

        while len(ranks) < 10:
            ranks.append(0)

    absences = set(neighbors.keys()) - set(ranked_.keys())
    for qname in absences:
        chosen = []
        j = 0
        while len(chosen) < 10 and j < len(neighbors[qname]):
            if neighbors[qname][j] not in chosen:
                chosen.append(neighbors[qname][j])
            j +=1
            
        while len(chosen) < 10:
            chosen.append(0)

        ranked_[qname] = chosen

    return ranked_


if __name__ == '__main__':
    neighbors = pickle.load(open(r'C:\Users\ASUS\Desktop\repositories\hum_to_find\neighbors.pkl', 'rb'))
    val_data = pickle.load(open(r'C:\Users\ASUS\Desktop\repositories\hum_to_find\crepe_freq\val_data.pkl', 'rb'))

    print(len(neighbors))

    for qname, nbs in neighbors.items():
        neighbors[qname] = [int(x) for x in neighbors[qname]]

    rs = predict_song(neighbors)

    print(len(rs))

    mrr = []
    for key in rs.keys():
        for tup in val_data:
            if key == tup[2]:
                if int(tup[0]) not in list(rs[key]):
                    mrr.append(0)
                else:
                    idx = list(rs[key]).index(int(tup[0])) +1
                    mrr.append(1/idx)
    print(np.mean(mrr))