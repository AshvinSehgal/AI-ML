class MinMaxScaler():
    def __init__():
        self.min = None
        self.max = None
    
    def fit(data):
        data = np.array(data)
        if(len(data.shape)==1):
            self.min = np.min(data)
            self.max = np.max(data)
        else:
            self.min = np.zeros(data.shape[1])
            self.max = np.zeros(data.shape[1])
            for col in data.shape[1]:
                self.min[col] = np.min(data[:,col])
                self.max[col] = np.max(data[:,col])
        return self
    
    def transform(data):
        data = np.array(data)
        if(len(data.shape)==1):
            data = (data-self.min)/(self.max-self.min)
        else:
            for col in data.shape[1]:
                data[:,col] = (data[:,col]-self.min[col])/(self.max[col]-self.min[col])
        return data
    
    def fit_transform(data):
        self.fit(data)
        return self.transform(data)
      
    def inverse_transform(data):
        data = np.array(data)
        if(len(data.shape)==1):
            data = (data*(self.max-self.min))+self.min
        else:
            for col in data.shape[1]:
                data[:,col] = (data[:,col]*(self.max[col]-self.min[col]))+self.min[col]
        return data
    
    def get_params():
        return self.min, self.max
    
    def set_params(min, max):
        self.min = min
        self.max = max
        return self
    
    def get_min():
        return self.min
    
    def get_max():
        return self.max
    
    def get_range():
        return self.max-self.min

class StandardScaler():
    def __init__():
        self.mean = None
        self.std = None
    
    def fit(data):
        data = np.array(data)
        if(len(data.shape)==1):
            self.mean = np.mean(data)
            self.std = np.std(data)
        else:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        return self
    
    def transform(data):
        data = np.array(data)
        if(len(data.shape)==1):
            data = (data-self.mean)/self.std
        else:
            for col in data.shape[1]:
                data[:,col] = (data[:,col]-self.mean[col])/self.std[col]
        return data
    
    def fit_transform(data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(data):
        data = np.array(data)
        if(len(data.shape)==1):
            data = (data*self.std)+self.mean
        else:
            for col in data.shape[1]:
                data[:,col] = (data[:,col]*self.std[col])+self.mean[col]
        return data
    
    def get_params():
        return self.mean, self.std
    
    def set_params(mean, std):
        self.mean = mean
        self.std = std
        return self
    
    def get_mean():
        return self.mean
    
    def get_std():
        return self.std