class landmark:
    landmark_locations = []
    landmark_sigs = []
    landmark_Qt = []
    
    def landmark(self,index):
        sig = self.landmark_sigs[index]
        l_pos = self.landmark_locations[index]
        Qt = self.landmark_Qt[index]
    
        return l_pos, sig, Qt
    
    def update_landmark(self, index, sig, l_pos, Qt):
        self.landmark_sigs[index] = sig
        self.landmark_locations[index] = l_pos
        self.landmark_Qt[index] = Qt
        
        
#landmark_location[Ct]
