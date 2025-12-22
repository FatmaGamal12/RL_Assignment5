import pickle


class ReplayBuffer:
    def __init__(self):
        # Stores tuples: (obs, action, reward, done)
        self.buffer = []

    def add(self, obs, action, reward, done):
        self.buffer.append((obs, action, reward, done))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)
        print(f"[ReplayBuffer] Saved {len(self.buffer)} transitions to {path}")

    def load(self, path):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)
        print(f"[ReplayBuffer] Loaded {len(self.buffer)} transitions from {path}")
