import pandas as pd
from tqdm import tqdm

from object_location_tuple import ObjectLocationTuple, combine_all_tuples


def get_object_locations_from_ascent() -> [ObjectLocationTuple]:
    data = pd.read_csv("../data/ascentpp.csv", delimiter=',', on_bad_lines='skip')
    filtered = data[data['relation'] == 'AtLocation']
    obj_loc_tuples = []

    for index, row in tqdm(filtered.iterrows(), 'Collecting data from Ascent++'):
        obj = row['subject']
        loc = row['object']
        obj_loc = ObjectLocationTuple(obj, loc, 'Ascent++')
        obj_loc_tuples.append(obj_loc)

    return combine_all_tuples(obj_loc_tuples)


if __name__ == '__main__':
    res = get_object_locations_from_ascent()
    for r in res:
        print(r)
