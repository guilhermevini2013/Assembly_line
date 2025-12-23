import os
import random
import time
import csv
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf


SIMULATION_INTERVAL = 5.0
CONVEYOR_SPEED = 1.0
PART_RATE = 0.9


class ConveyorBelt:
    def __init__(self):
        self.speed = CONVEYOR_SPEED
        self.motor_current = 0.0
        self.vibration = 0.0
        self.temperature = 30.0

    def update_sensors(self):
        load = 3
        self.motor_current = 2.0 + load * 0.8 + random.uniform(-0.1, 0.1)
        self.vibration = 0.2 + load * 0.1 + random.uniform(0, 0.05)
        self.temperature += self.motor_current * 0.02


class Simulator:
    def __init__(self):
        self.conveyor = ConveyorBelt()
        self.dataset = []

    def generate_part(self):
        path_list = ["../dataset/test/Anomaly", "../dataset/test/Normal"]
        current_path = random.choice(path_list)
        files = os.listdir(current_path)
        return current_path + "/" + random.choice(files)

    def save_dataset(self):
        with open("dataset.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.dataset[0].keys()
            )
            writer.writeheader()
            writer.writerows(self.dataset)

    def evaluate(self, image):
        model = tf.keras.models.load_model("../cnn/weights/weightV1.h5")
        image = cv2.resize(image, (128, 128))
        image = image / 255
        image = image.reshape(-1, 128, 128, 3)

        y_predict = model.predict(image)
        predicted_class = np.argmax(y_predict)

        return predicted_class, y_predict[0][0]

    def run(self):
        while True:
            part_path = self.generate_part()
            part_image = cv2.imread(part_path)

            self.conveyor.update_sensors()

            sensors = {
                "speed": self.conveyor.speed,
                "motor_current": self.conveyor.motor_current,
                "vibration": self.conveyor.vibration,
                "temperature": self.conveyor.temperature
            }

            predicted_class, score = self.evaluate(part_image)

            record = {
                "timestamp": datetime.now().isoformat(),
                **sensors,
                "anomaly_score": float(score),
                "is_accepted": bool(predicted_class)
            }

            self.dataset.append(record)

            print(record)

            time.sleep(SIMULATION_INTERVAL)


if __name__ == "__main__":
    sim = Simulator()
    sim.run()
