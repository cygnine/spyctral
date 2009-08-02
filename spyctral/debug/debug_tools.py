"""
Tools for debugging spyctral
"""

def default_validator(data):
    """ 
    The default validator is always True
    """
    return True

def default_data(params):
    return None

class ValidationTest:

    def __str__(self):
        return "%s \n    " % self.description + str(self.parameters)

    def __init__(self,description=None, parameters=dict(),
            data_generator=default_data, validator=default_validator):
        self.result = None
        self.description = description
        self.parameters = parameters
        self.validator = validator
        self.data_generator = data_generator

    def run_test(self):
        self.result = self.validator(self.data_generator(**self.parameters),
                **self.parameters)

class ValidationContainer:
    """
    A container for validation tests
    """

    def __init__(self):
        self.N = 0
        self.validation_tests = list()
        self.failed_test_indices = list()

    def get_results(self):
        return [test.result for test in self.validation_tests]

    def get_validators(self):
        return [test.validators for test in self.validation_tests]

    def get_data_generators(self):
        return [test.data_generator for test in self.validation_tests]

    def get_parameters(self):
        return [test.parameters for test in self.validation_tests]

    def add_test(self,description=None, parameters=dict(),
            validator=default_validator,data_generator=default_data):
        test = ValidationTest(**kwargs)
        self.validation_tests.append(test)
        self.N += 1

    def add_tests(self,test_list):
        for test in test_list:
            self.validation_tests.append(test)
            self.N += 1

    def run_tests(self):
        self.failed_test_indices = list()
        for f_test,n in zip(self.validation_tests,range(self.N)):
            f_test.run_test()
            if not f_test.result:
                self.failed_test_indices.append(n)

        if len(self.failed_test_indices)>0:
            print "Some tests failed"
        else:
            print "All tests passed"

    def show_failed_tests(self):
        if len(self.failed_test_indices)==0:
            print "All tests passed"
        else:
            for n in self.failed_tests:
                print self.validation_tests[n]

    def show_passed_tests(self):
        for test in self.validation_tests:
            if test.result:
                print test

    def extend(self,other):
        self.add_tests(other.validation_tests)

    def append(self,vtest):
        self.add_tests([vtest])
