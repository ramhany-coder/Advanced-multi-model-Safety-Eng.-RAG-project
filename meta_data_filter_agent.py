class meta_data_filter_agent:
    def __init__(self, meta_data):
        self.meta_data = meta_data

    def filter_meta_data(self, filter_criteria):
        """
        Filters the meta data based on the provided criteria.

        :param filter_criteria: A dictionary containing the filtering criteria.
        :return: Filtered meta data.
        """
        filtered_data = []
        for item in self.meta_data:
            if all(item.get(key) == value for key, value in filter_criteria.items()):
                filtered_data.append(item)
        return filtered_data