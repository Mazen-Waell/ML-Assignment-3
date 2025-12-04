import numpy as np
class Node :
    def __init__(self,feature_index,threshold,left,right,value):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

class DecisionTreeClassifier :
    def __init__(self,max_depth,min_samples_split):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None

    def fit (self,X,y):
        self.root=self.build_tree(X,y,depth=0)
    
    def predict(self,X):
        prediciton= []
        for x in X :
            pred=self.predict_sample(x)
            prediciton.append(pred)
        return prediciton    


    def predict_sample(self,x) :
        node=self.root 
        while(node.value is None) :
            if x[node.feature_index] <= node.threshold :
                node=node.left
            else:
                node=node.right   
        return node.value         

        

    def entropy (self,y):
        hist=np.bincount(y)
        ps = hist / len(y)
        return - sum ( p * np.log2(p) for p in ps if p > 0  )

    def information_gain(self,X,y,feature_index,threshold) :
        parent_entropy=self.entropy(y)

        left_index = X[:, feature_index] <= threshold
        right_index = X[:, feature_index] > threshold

        y_left = y[left_index]
        y_right = y[right_index]

        n_l, n_r = len(y_left), len(y_right)

        if n_l == 0 or n_r == 0:
            return 0  

        n=len(y)
        e_l, e_r = self.entropy(y_left), self.entropy(y_right)
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r



        IG=parent_entropy-child_entropy
        return IG
 

    def majority_class(self,y) :
      return max(set(y),key=y.tolist().count)   

    def build_tree(self,X,y,depth) :
        if (depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split ):
            return Node(
                feature_index=None,
                threshold=None,
                left=None,
                right=None,
                value=self.majority_class(y)
            )
        
       # best split
        best_feature_index= None
        best_threshold=None
        best_IG=-1

        for feature_index in range(X.shape[1]) :
            thredsholds = np.unique(X[:,feature_index])
            for threshold in thredsholds :
                IG=self.information_gain(X,y,feature_index,threshold)
                if IG > best_IG :
                    best_IG = IG
                    best_feature_index=feature_index
                    best_threshold=threshold

        if best_IG <= 0:
               return Node(
                feature_index=None,
                threshold=None,
                left=None,
                right=None,
                value=self.majority_class(y)
                )

        left_index = X[:, best_feature_index] <= best_threshold
        right_index = X[:, best_feature_index] > best_threshold

        left_node = self.build_tree(X[left_index], y[left_index], depth + 1)
        right_node = self.build_tree(X[right_index], y[right_index], depth + 1)

        return Node(
                feature_index=best_feature_index,
                threshold=best_threshold,
                left=left_node,
                right=right_node,
                value=None
            )