# Read HBML style ecog.mat files
def read_ecog(self, ecogFile, extra_neuraldata=None, analog_channels=None, read_all=True):
    df = labelfile['cleaned']

    # Read in the file
    ecog = read_mat(ecogFile, variable_names=['ecog'])
    ecog = ecog['ecog']
    self.raw_data['orig'] = ecog

    # Get general variables
    task = ecog['task']
    self.session_description = task
    block = os.path.basename(ecog['filename'])
    self.session_id = block

    # Set fieldtrip variable
    ftrip = ecog['ftrip']
    fs = ftrip['fsample']
    neuraldata = ftrip['trial']
    self.raw_data['neural']['labels'] = ftrip['label']
    self.raw_data['neural']['data'] = neuraldata
    self.raw_data['neural']['fs'] = fs