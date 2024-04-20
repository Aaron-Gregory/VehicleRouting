from ast import literal_eval
import sys
import pandas as pd

MAX_ROUTE_LENGTH = 12 * 60


def cost(routes, dataframe):
    """
    Compute the cost of the given routes, with pickup and dropoff location data stored in the dataframe
    """
    number_of_drivers = len(routes)
    total_number_of_driven_minutes = sum(
        [get_route_length(r, dataframe) for r in routes]
    )

    return number_of_drivers * 500 + total_number_of_driven_minutes


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_route_length(route, dataframe):
    """
    Returns the length of a path starting and ending at the origin and
    going through every delivery in the route in order.
    """
    total_length = dist(dataframe.dropoff.loc[route[-1]], (0, 0))
    for i in range(len(route)):
        coming_from = (0, 0)
        if i != 0:
            coming_from = dataframe.dropoff.loc[route[i - 1]]

        pickup_point = dataframe.pickup.loc[route[i]]
        dropoff_point = dataframe.dropoff.loc[route[i]]

        dist_to_pickup = dist(coming_from, pickup_point)
        dist_to_dropoff = dist(pickup_point, dropoff_point)
        total_length += dist_to_pickup + dist_to_dropoff

    return total_length


def vrp_greedy_solution(data):
    """
    Computes a greedy solution to the vehicle routing problem. Builds routes to adding
    the closest pickup after each dropoff, so long as max route length is not exceeded.
    """
    remaining_points = data.copy()

    active_route = []
    current_loc = (0, 0)
    routes = [active_route]
    while len(remaining_points) > 0:
        # find next point
        remaining_points["dists"] = remaining_points.pickup.apply(
            lambda x: dist(current_loc, x)
        )
        closest_idx = remaining_points.dists.idxmin()

        # if the added point makes the route too long, start a new one
        if get_route_length(active_route + [closest_idx], data) > MAX_ROUTE_LENGTH:
            active_route = []
            current_loc = (0, 0)
            routes.append(active_route)
        else:
            # if the route is not too long, add the point to it and remove the row from consideration
            active_route.append(closest_idx)
            remaining_points.drop(index=closest_idx, inplace=True)

    return routes


if __name__ == "__main__":
    # get the path to the problem specification file
    path_to_file = sys.argv[1]

    # load and parse the data
    data = pd.read_csv(path_to_file, delimiter=" ")
    data = data.set_index("loadNumber")
    data.pickup = data.pickup.apply(literal_eval)
    data.dropoff = data.dropoff.apply(literal_eval)

    # compute and print the solution
    solution = vrp_greedy_solution(data)

    for route in solution:
        print(route)
