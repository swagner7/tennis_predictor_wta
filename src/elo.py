from collections import defaultdict

class EloTracker:
    def __init__(self, k=32, k_new=40, new_threshold=30):
        self.k = k
        self.k_new = k_new
        self.new_threshold = new_threshold
        self.global_elo = defaultdict(lambda: 1500)
        self.context_elo = defaultdict(lambda: defaultdict(lambda: 1500))
        self.match_count = defaultdict(int)

    def get_k(self, player):
        return self.k_new if self.match_count[player] < self.new_threshold else self.k

    def get_elo(self, player, context):
        return self.global_elo[player], self.context_elo[context][player]

    def update_elo(self, p1, p2, winner, context):
        # Global Elo update
        Ra, Rb = self.global_elo[p1], self.global_elo[p2]
        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Eb = 1 - Ea
        Sa, Sb = (1, 0) if winner == p1 else (0, 1)
        k1, k2 = self.get_k(p1), self.get_k(p2)

        self.global_elo[p1] += k1 * (Sa - Ea)
        self.global_elo[p2] += k2 * (Sb - Eb)

        # Context-specific Elo update (surface, series, round, etc.)
        Rc1 = self.context_elo[context][p1]
        Rc2 = self.context_elo[context][p2]
        Ec1 = 1 / (1 + 10 ** ((Rc2 - Rc1) / 400))
        Ec2 = 1 - Ec1

        self.context_elo[context][p1] += k1 * (Sa - Ec1)
        self.context_elo[context][p2] += k2 * (Sb - Ec2)

        # Record match counts
        self.match_count[p1] += 1
        self.match_count[p2] += 1
