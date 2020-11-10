from tlorem import TLOREM

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    random_state = 0

    binary = True

    if binary:
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    else:
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')

    class_name = 'class'
    class_values = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in
                    newsgroups_train.target_names]

    if not binary:
        class_values[3] = 'pc.hardware'
        class_values[4] = 'mac.hardware'

    vectorizer = TfidfVectorizer(lowercase=False)  # Convert a collection of raw documents to a matrix of TF-IDF features
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    X_test = vectorizer.transform(newsgroups_test.data)
    Y_train = newsgroups_train.target
    Y_test = newsgroups_test.target

    K = newsgroups_train.data

    bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
    # bb = SVC(random_state=random_state)
    # bb = MLPClassifier(random_state=random_state)
    bb.fit(X_train, Y_train)

    def bb_predict(texts):
        Xtfidf = vectorizer.transform(texts)
        return bb.predict(Xtfidf)

    i2e = 2
    text = newsgroups_test.data[i2e]
    x_text = X_test[i2e]

    print('text = "%s"' % text)
    print('')

    bb_outcome = bb_predict([text])[0]
    bb_outcome_str = class_values[bb_outcome]
    print('bb(x) = { %s }' % bb_outcome_str)
    print('')

    explainer = TLOREM(K, bb_predict, class_name, class_values, neigh_type='lime', size=1000, ocr=0.1, bow=True,
                       split_expression=r'\W+', kernel_width=None, kernel=None, random_state=0, verbose=True)

    exp = explainer.explain_instance(text, num_samples=1000, use_weights=True, metric='cosine')

    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    # for crule in exp.crules:
    #     print(crule)
    # print(exp.bb_pred, exp.dt_pred, exp.fidelity)
    print('')

    print(exp.get_text_rule())
    print('')

    for text_crule in exp.get_text_counterfactuals():
        print(text_crule)
    print('')


if __name__ == "__main__":
    main()