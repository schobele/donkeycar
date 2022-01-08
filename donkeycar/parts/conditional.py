class ConditionalPart(object):

    def __init__(self):

        print("conditions: -1: left")
        print("conditions: 0: straight")
        print("conditions: 1: right")
        conditions = [-1, 0, 1]
        self.conditions = conditions
        self.active_condition = 0

    def left(self):
        if self.active_condition != 0:
            self.active_condition = 0
        else:
            self.active_condition = 1
        print("conditional command set:",
              self.conditions[self.active_condition])

    def right(self):
        if self.active_condition != 2:
            self.active_condition = 2
        else:
            self.active_condition = 1
        print("conditional command set:",
              self.conditions[self.active_condition])

    def straight(self):
        self.active_condition = 0
        print("conditional command set:",
              self.conditions[self.active_condition])

    def run(self):
        return self.active_condition, self.conditions[self.active_condition]

    def shutdown(self):
        pass
