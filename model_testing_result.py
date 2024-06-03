class ModelTestingResult:
    #def __init__(self, model_name, accuracy, precision, recall, f1_score, confusion_matrix):
    def __init__(self, loss, accuracy, mismatched_label_stats, mismatched_label_images):
        self.loss = loss
        self.accuracy = accuracy
        self.mismatched_label_stats = mismatched_label_stats
        self.mismatched_label_images = mismatched_label_images
        # self.precision = precision
        # self.recall = recall
        # self.f1_score = f1_score
        # self.confusion_matrix = confusion_matrix

    def __str__(self):
        return f"Loss: {self.loss}\n" \
               f"Accuracy: {self.accuracy}\n" \
                f"Mismatched Label Stats: {self.mismatched_label_stats}\n" \
                f"Mismatched Label Images: {self.mismatched_label_images}\n"
            #    f"Precision: {self.precision}\n" \
            #    f"Recall: {self.recall}\n" \
            #    f"F1 Score: {self.f1_score}\n" \
            #    f"Confusion Matrix: {self.confusion_matrix}\n"

    def __repr__(self):
        return str(self)
    