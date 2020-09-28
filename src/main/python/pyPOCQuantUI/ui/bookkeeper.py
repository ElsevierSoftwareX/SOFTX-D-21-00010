

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
        self.rect = self.num_timepoints * [None]
        self.line = self.num_timepoints * [None]

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

    def getCurrentLine(self):
        """
        Get CompositeLine for current timepoint (or None if it does not exist).
        """
        return self.line[self.timepoint]

    def addLine(self, line, indices=None):
        """
        Add a CompositeLine at current timepoint.
        """
        if not indices:
            self.line[self.timepoint] = line
        else:
            for idx in indices:
                self.line[idx] = line

    def removeLine(self):
        """
        Removes the CompositeLine from the bookKeeper.
        """
        self.line = self.num_timepoints * [None]

    def addHoughRect(self, rect, indices=None):
        """
        Add a hough rect at current timepoint.
        """
        if not indices:
            self.rect[self.timepoint] = rect
        else:
            for idx in indices:
                self.rect[idx] = rect

    def getCurrentHoughRect(self):
        """
        Get hough rect for current timepoint (or None if it does not exist).
        """
        return self.rect[self.timepoint]

