import numpy as np
import matplotlib.pyplot as plt

class ArtMarket:
    def __init__(self, players=5):
        self.num_players = players
        self.initialize_players()
        self.creation_time = {
            'L': 1,
            'M': 3,
            'H': 5
        }
        self.price_ranges = {
            'L': (1, 10),
            'M': (10, 50),
            'H': (50, 100)
        }
        
    def initialize_players(self):
        self.players = [
            {
                'Tokens': 1000,
                'Artworks': [],
                'Own artworks': [],
                'Money weightage': np.random.rand(),
                'Artwork weightage': None
            }
            for i in range(self.num_players)
        ]
        for i in range(self.num_players):
            self.players[i]['Artwork weightage'] = 1 - self.players[i]['Money weightage']
    
    def create_artwork(self, player, artwork_quality):
        artwork = {
            'Quality': artwork_quality,
            'Price': np.random.randint(self.price_ranges[artwork_level][0], self.price_ranges[artwork_level][1]),
            'Creator': player,
            'Owner': None,
        }
        return artwork
            
    def upload_artworks(self, player, num_artworks, artwork_quality):
        for i in range(num_artworks):
            artwork = self.create_artwork(player, artwork_quality[i])
            player['Artworks'].append(artwork)
            player['Own artworks'].append(artwork)
            self.listings.append(artwork)
    
    def buy_artworks(self, player, quality):
        artwork = self.create_artwork(player, quality)
        player['Artworks'].append(artwork)
        player['Tokens'] -= artwork['Price']
        return True
        