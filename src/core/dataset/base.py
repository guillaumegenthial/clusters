from core.utils.general import check_dir, pickle_dump, Progbar

class DatasetBase(object):
    """
    Base class for dataset generator
    """
    def __init__(self):
        self.length = None
        
    def __iter__(self):
        """
        Must yield a tuple (cluster, no of event)
        """
        raise NotImplementedError

    def __len__(self):
        if self.length is None:
            print "WARNING : iterating over all corpus to build length"
            count = 0
            for _ in self:
                count += 1
            self.length= count
            print "- done."
            
        return self.length

    def export_one_file_per_event(self, path):
        """
        Iterates over dataset and dumps one file for each event
        """
        prog = Progbar(target=self.max_iter)
        check_dir(path)
        event_no_ref = 0
        event = []
        for cluster, event_no in self:
            if event_no != event_no_ref:
                export_path = path+"/event_{}_nclusters_{}.npy".format(event_no_ref, len(event))
                pickle_dump(event, export_path, False)
                event_no_ref = event_no
                event = []
                prog.update(event_no+1)
            event += [cluster]

    def get_data(self):
        """
        Iterates over the data and stores everything in a list
        Can take a lot of memory
        Returns:
            a list of dict
        """
        data = []
        for d_, _ in self:
            data.append(d_)
        return data
