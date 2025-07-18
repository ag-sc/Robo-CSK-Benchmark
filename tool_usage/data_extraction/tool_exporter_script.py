# This script has been used to create an extract from the complete 'Narrative Objects' KG To execute it, follow the
# instructions here: https://github.com/ease-crc/ease_lexical_resources/wiki/Querying-DFL-with-a-DL-reasoner The
# resulting JSON file can be found in tool_usage/data and is further processed through the
# narrative_objects_extractor.py

import json

import dfl.dlquery as dl
from tqdm import tqdm

res = {}
dl.buildCache()
tools = dl.whatSubclasses("dfl:tool.n.wn.artifact")
for tool in tqdm(tools, 'Extracting tools from Narrative Objects'):
    disp = dl.whatDispositionsDoesObjectHave(tool)
    proc_tool = tool.replace(".n.wn.artifact", "").replace("dfl:", "").split("..")[0].strip()
    proc_disp = [d.split(".v")[0].replace("dfl:", "").strip() for d in disp]
    res[proc_tool] = proc_disp

with open("narrative_tools.json", "w") as fp:
    json.dump(res, fp)
