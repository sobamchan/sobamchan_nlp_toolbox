import xml.etree.ElementTree as ET

class AGCorpus:

    def __init__(self, datapath):
        self.tree = ET.parse(datapath)
        self.titles = None
        self.categories = None
        self.descs = None

    def get_data(self):
        titles = [ title_ele.text for title_ele in self.tree.findall('title') ]
        categories = [ category_ele.text for category_ele in self.tree.findall('category') ]
        descs = [ desc_ele.text for desc_ele in self.tree.findall('description') ]

        self.titles = titles
        self.categories = categories
        self.descs = descs

        classes = {'Business': 0, 'Entertainment': 1, 'World': 2, 'Sports': 3}
        X = []
        T = []
        for category, desc in zip(categories, descs):
            if category in classes.keys() and desc is not None:
                X.append(desc)
                T.append(classes[category])

        return T, X
