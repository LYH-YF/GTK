import json


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""
    return data


def write_json_data(data, path):  #write data into a json file
    with open(path, mode="w+", encoding="utf-8") as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)
    f.close()


def load_json_data(path):  #load data from a json file
    f = open(path, mode="r", encoding="utf-8")
    json_data = json.load(f)
    return json_data