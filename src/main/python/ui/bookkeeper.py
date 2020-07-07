import os
from .compositePolygon import CompositePolygon


class BookKeeper:

    def __init__(self):
        """
        Constructor.
        """

        self.timepoint = 0
        self.num_timepoints = self.timepoint + 1
        self.images = self.num_timepoints * [None]
        self.stripPolygon = self.num_timepoints * [None]
        self.sensorPolygon = self.num_timepoints * [None]

    # def getCurrentTimepoint(self):
    #     """
    #       Get the current timepoint.
    #     """
    #     return self.timepoint

    # def getCurrentCompositeLine(self):
    #     """
    #     Get CompositeLine for current timepoint (or None if it does not exist).
    #     """
    #     return self.compositeLines[self.timepoint]

    # def addCompositeLine(self, compositeLine, indices=None):
    #     """
    #     Add a CompositeLine at current timepoint.
    #     """
    #     if not indices:
    #         self.compositeLines[self.timepoint] = compositeLine
    #     else:
    #         for idx in indices:
    #             pos = compositeLine.pos
    #             self.compositeLines[idx] = CompositeLine(pos)

    def getCurrentStripPolygon(self):
        """
        Get CompositePolygon for current timepoint (or None if it does not exist).
        """
        return self.stripPolygon[self.timepoint]

    def addStripPolygon(self, stripPolygon, indices=None):
        """
        Add a CompositePolygon at current timepoint.
        """
        if not indices:
            self.stripPolygon[self.timepoint] = stripPolygon
        else:
            for idx in indices:
                self.stripPolygon[idx] = stripPolygon

    def getCurrentSensorPolygon(self):
        """
        Get CompositePolygon for current timepoint (or None if it does not exist).
        """
        return self.sensorPolygon[self.timepoint]

    def addSensorPolygon(self, sensorPolygon, indices=None):
        """
        Add a CompositePolygon at current timepoint.
        """
        if not indices:
            self.sensorPolygon[self.timepoint] = sensorPolygon
        else:
            for idx in indices:
                self.sensorPolygon[idx] = sensorPolygon

