import googlemaps


def init():
    return Routes()


class Routes:

    gmaps_key = None

    def __init__(self):
        self.gmaps_key = googlemaps.Client(key='AIzaSyDpKjFv9Nq8B6rDerR1W-EZyHZm-cHSNFk')

    def coords(self):
        return Coords(self.gmaps_key)


class Coords:

    def __init__(self, key):
        self.gmaps_key = key

    def get_coordinates(self, position, as_list=False, set_city=None):

        if type(position) is str:

            if set_city is not None:
                position = position + ' ' + str(set_city)

            try:
                geocode_result = self.gmaps_key.geocode(position)

                lat = geocode_result[0]["geometry"]["location"]["lat"]
                lng = geocode_result[0]["geometry"]["location"]["lng"]

                if as_list:
                    coords = [lat, lng]
                else:
                    coords = {
                        "lat": lat,
                        "lng": lng
                    }

                return coords

            except(ValueError, Exception):
                raise

        else:
            return position

    def get_address(self, coordinates, set_city=None):

        try:

            reverse = self.gmaps_key.reverse_geocode(coordinates)[0]['address_components']
            reverse = str(reverse[1]['short_name']) + ' #' + str(reverse[0]['short_name'])

            if set_city is not None:

                return reverse + ' ' + set_city
            else:

                return reverse

        except(ValueError, Exception):
            raise
