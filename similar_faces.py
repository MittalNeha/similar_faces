from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
import numpy as np


class FaceSimilarity:
    def __init__(self, df):
        self.trainset_len = 1000
        self.train_corpus = []
        self.train_image_names = []
        self.nn_tree = None

        self.model = self.create_w2v_model(df)
        # initialize the model for NearestNeighbor search
        self.nn_init()

    def create_w2v_model(self, df):
        # pick the columns with attribute value ==1 and use this to form the training set
        t_data = df.apply(lambda row: row[row == 1].index, axis=1)
        self.train_image_names = list(df.loc[:, 'image_id'])
        self.train_corpus = [list(row.values) for index, row in t_data.items()]

        # train word2vec model
        model = Word2Vec(window=10, sg=1, hs=0,
                         negative=10,  # for negative sampling
                         alpha=0.03, min_alpha=0.0007,
                         seed=14)
        model.build_vocab(self.train_corpus, progress_per=200)
        model.train(self.train_corpus, total_examples=model.corpus_count,
                    epochs=10, report_delay=1)

        print("Created a Word2Vec model")

        return model

    # mean of all the face attributes
    def get_embeddings(self, attr, model):
        face_vec = []
        for key in attr:
            face_vec.append(model[key])

        return np.mean(face_vec, axis=0)

    def nn_init(self):
        # So each face in the train_set can be represented by a vector. lets take the  mean.
        face_vector = {self.train_image_names[i]: self.get_embeddings(self.train_corpus[i], self.model) for i in
                       range(self.trainset_len)}
        a = list(face_vector.values())
        b = np.array(a)

        print("Train NearestNeighbors")
        self.nn_tree = NearestNeighbors(
            n_neighbors=30, algorithm='ball_tree')
        self.nn_tree.fit(b)

    def get_attr(self, test_attr, groups):
        attr = []
        for idx, gp in enumerate(test_attr.values()):
            A = ((gp >= 0.52).nonzero(as_tuple=True)[1]).tolist()
            if len(A) > 0:
                attr.extend([groups[idx][at] for at in A])
        print (attr)
        return attr

    '''
    Takes the image attributes and gives out 'n' closest faces to the given image
    '''

    def similar_faces(self, image_attr, groups, n=5):
        image_attr = self.get_attr(image_attr, groups)
        v = self.get_embeddings(image_attr, self.model)
        nbrs = self.nn_tree.radius_neighbors([v])
        nearest_face_idx = nbrs[1][0][:n]
        print(nearest_face_idx)

        sim_images = [self.train_image_names[face] for face in nearest_face_idx]
        return sim_images
