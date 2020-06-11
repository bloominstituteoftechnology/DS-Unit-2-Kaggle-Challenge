wrangle.py
Class Wrangler:
    """ collection of parameters and methods for manipulating dataframe in 
    a documented and reproducible way"""

    transforms = [] #list of operaations to perfom, as transformation method names - order?
 
    droplist = {} #dict of column names and reasons for dropping
    def drop((colname, reason='none')):
        """take tuple of strings, column name to add to droplist, and reason"""
        droplist[colname] = reason      #add to droplist

##column transformations 
    def lcase_all(self): 
        """if invoked replace all string column with lcase version"""
        transforms.append['lcase_all']
    