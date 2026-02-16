from nltk.stem import WordNetLemmatizer

from ranked_location import RankedLocation
from utils import check_if_household_location, check_if_household_object

wnl = WordNetLemmatizer()


class ObjectLocationTuple:
    def __init__(self, obj: str, initial_loc: str, source: str, should_check=True):
        proc_obj = preprocess_string(obj)
        proc_loc = preprocess_string(initial_loc)
        if should_check:
            self._is_correct = check_if_household_object(proc_obj) & check_if_household_location(proc_loc)
        else:
            self._is_correct = True
        self._object = proc_obj
        loc = RankedLocation(proc_loc, [source])
        self._locations = [loc]

    def __str__(self):
        return f'[{str(self.verify()).upper()}] {self._object} located at/in: {self._locations}'

    def verify(self) -> bool:
        return self._is_correct

    def get_object(self) -> str:
        return self._object

    def get_locations(self) -> [RankedLocation]:
        return self._locations

    def add_location(self, location: RankedLocation):
        for loc in self._locations:
            if loc.get_location() == location.get_location():
                for source in location.get_sources():
                    loc.add_source(source)
                return

        self._locations.append(location)

    def remove_location(self, locations: [RankedLocation]):
        len_bef = len(self._locations)
        self._locations = [l for l in self._locations if l not in locations]
        if len(self._locations) == 0:
            self._is_correct = False
        assert len_bef == len(self._locations) + len(locations)

    def combine_tuples(self, tup: 'ObjectLocationTuple'):
        if not tup.verify():
            return

        if tup.get_object() != self.get_object():
            return

        for loc in tup.get_locations():
            self.add_location(loc)

        tup._is_correct = False

    def to_dict(self):
        locs = {}
        rank = 1
        sorted_locs = sorted(self._locations)
        for l in sorted_locs:
            locs[rank] = (l.get_location(), l.get_trust())
            rank += 1

        return {
            'Object': self.get_object(),
            'Locations': locs
        }


def preprocess_string(word: str) -> str:
    processed = word.lower().replace('the', '').replace('_', ' ').strip()
    return wnl.lemmatize(processed)


def combine_all_tuples(tuples: [ObjectLocationTuple]) -> [ObjectLocationTuple]:
    combined = {}
    for tup in [t for t in tuples if t.verify]:
        obj = tup.get_object()
        if obj in combined:
            combined[obj].combine_tuples(tup)
        else:
            combined[obj] = tup
    return list(combined.values())
