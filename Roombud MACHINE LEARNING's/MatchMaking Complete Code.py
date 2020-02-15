import Algorithmia
from nltk.stem import SnowballStemmer
import copy
import random, string

example_input = {
    "group1": [
      {
          "name": "Vincent the Conquerer",
          "interests": [
              "reading",
              "running",
              "chilling",
              "coding",
              "seattle",
              "coffee",
              "tea",
              "bilingual",
              "food",
              "arrested development",
              "the office",
              "parc and rec",
              "rick and morty"
          ],
          "values": [
              "humanism"
          ],
          "age": "22",
          "coordinates": {
              "lat": 47.599088077746394,
              "long": -122.3339125374332
          }
      },
      {
          "name": "Paul the Extrovert",
          "interests": [
              "hiking",
              "skiing",
              "coffee",
              "traveling"
          ],
          "values": [
              "adventure"
          ],
          "age": "26",
          "coordinates": {
              "lat": 47.599088077746394,
              "long": -122.3339125374332
          }
      },
      {
          "name": "Tom the Family Guy",
          "interests": [
              "reading",
              "writing",
              "coffee",
              "binge watching",
              "netflix"
          ],
          "values": [
              "family"
          ],
          "age": "32",
          "coordinates": {
              "lat": 47.599088077746394,
              "long": -122.3339125374332
          }
      }
    ],
    "group2": [
      {
          "name": "Julia the Jukebox",
          "interests": [
              "music",
              "rock",
              "coffee",
              "guitar hero"
          ],
          "values": [
              "individuality"
          ],
          "age": "22",
          "coordinates": {
              "lat": 47.62446091996251,
              "long": -122.32016064226627
          }
      },
      {
          "name": "Chelsea the Bookworm",
          "interests": [
              "reading",
              "writing",
              "classics",
              "coffee",
              "walking"
          ],
          "values": [
              "family",
              "love"
          ],
          "age": "26",
          "coordinates": {
              "lat": 47.62446091996251,
              "long": -122.32016064226627
      }
      },
      {
          "name": "Ana the Artist",
          "interests": [
              "drawing",
              "art",
              "music",
              "classical music",
              "tea",
              "running"
          ],
          "values": [
              "post-modernism",
              "beauty"
          ],
          "age": "32",
          "coordinates": {
              "lat": 47.62446091996251,
              "long": -122.32016064226627
          }
      }
    ]
}

client = Algorithmia.client('simF9pfoGFXY1Sk3VPvE236nIIg1')

class AlgorithmError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

def apply(input):
    # default weights for the scoring function
    default_weights = {
        "interests": 1.0,
        "values": 5.0,
        "age": 0.5,
        "coordinates": 0.005
    }
    # overwrite the weights if given by user
    if "scoring_weights" in input:
        weights = overwriteWeights(default_weights, input["scoring_weights"])
    else:
        weights = default_weights
    
    # get the input and do some checking
    validateInput(input)
    
    stabletraitsInput = {"optimal": {}, "pessimal": {}}
    
    male_scoring_list = {}
    female_scoring_list = {}
    # create a preference list for each individual using the scoring function
    for maleObject in input["group1"]:
        male_scoring_list[maleObject["name"]] = {}
        for femalObject in input["group2"]:
            score = scoring_function(weights, maleObject, femalObject)
            male_scoring_list[maleObject["name"]][femalObject["name"]] = score
    
    for femalObject in input["group2"]:
        female_scoring_list[femalObject["name"]] = {}
        for maleObject in input["group1"]:
            score = scoring_function(weights, femalObject, maleObject)
            female_scoring_list[femalObject["name"]][maleObject["name"]] = score
    
    tmp_male_scoring_list = copy.deepcopy(male_scoring_list)
    tmp_female_scoring_list = copy.deepcopy(female_scoring_list)
    
    print("One To Matching", male_scoring_list)

    # map & sort the scoring lists into a format that preserves the order of the objects
    for male in tmp_male_scoring_list:
        # map into a sortable format
        male_scoring_list[male] = list(map(lambda x: {"name": x, "similarity": male_scoring_list[male][x]}, male_scoring_list[male]))
        # sort the preference list
        male_scoring_list[male] = sorted(male_scoring_list[male], key=lambda k: k['similarity'], reverse=True)
        # remove the similarity scores from the preference lists
        male_scoring_list[male] = list(map(lambda x: x["name"], male_scoring_list[male]))
    
    for female in tmp_female_scoring_list:
        # map into a sortable format
        female_scoring_list[female] = list(map(lambda x: {"name": x, "similarity": female_scoring_list[female][x]}, female_scoring_list[female]))
        # sort the preference list
        female_scoring_list[female] = sorted(female_scoring_list[female], key=lambda k: k['similarity'], reverse=True)
        # remove the similarity scores from the preference lists
        female_scoring_list[female] = list(map(lambda x: x["name"], female_scoring_list[female]))
    
    # if one group has a larger preference list, add null characters to the end of the list
    # this is to ensure that the stable traits algorithm works properly
    group_difference = len(male_scoring_list) - len(female_scoring_list)
    null_people = []
    if group_difference == 0.0:
        # create stable pairs using the given preference lists with the stable traits algorithm
        stable_traits_input = {
            "optimal": male_scoring_list,
            "pessimal": female_scoring_list
        }
    else:
        if group_difference > 0:
            for i in range(group_difference):
                null_female = randomword(20)
                null_people.append(null_female)
                female_scoring_list[null_female] = []
                for male in male_scoring_list:
                    male_scoring_list[male].append(null_female)
                    female_scoring_list[null_female].append(male)
        elif group_difference < 0:
            for i in range(group_difference):
                null_male = randomword(20)
                null_people.append(null_male)
                male_scoring_list[null_female] = []
                for female in female_scoring_list:
                    female_scoring_list[female].append(null_male)
                    male_scoring_list[null_female].append(female)
                    
        # create stable pairs using the given preference lists with the stable traits algorithm
        stable_traits_input = {
            "optimal": male_scoring_list,
            "pessimal": female_scoring_list
        }
    
    stable_traitss = client.algo("matching/StableMarriageAlgorithm").pipe(stable_traits_input).result["matches"]
    
    if group_difference == 0.0:
        return stable_traitss
    elif group_difference > 0:
        tmp = copy.deepcopy(stable_traitss)
        stable_traitss = dict((v,k) for k,v in tmp.iteritems())
        for person in null_people:
            stable_traitss.pop(person)
        return stable_traitss
    elif group_difference < 0:
        for person in null_people:
            stable_traitss.pop(person)
        return stable_traitss

def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

def overwriteWeights(default, new):
    rVal = default
    
    if "interests" in new:
        rVal["interests"] = float(new["interests"])
    if "values" in new:
        rVal["values"] = float(new["values"])
    if "age" in new:
        rVal["age"] = float(new["age"])
    if "coordinates" in new:
        rVal["coordinates"] = float(new["coordinates"])
    
    return rVal

def scoring_function(weights, person1, person2):
    # returns a score that gives the similarity between 2 people
    # scoring function:
    #   +add for each interest * weight
    #   +add for each value * weight
    #   -subtract age difference * weight
    #   -subtract location difference * weight
    ss = SnowballStemmer("english")
    score = 0.0
    
    interest_list1 = person1["interests"]
    interest_list2 = person2["interests"]
    
    # compare similar interests
    for interest1 in interest_list1:
        for interest2 in interest_list2:
            stem1 = ss.stem(interest1.lower())
            stem2 = ss.stem(interest2.lower())
            
            if stem1 == stem2:
                score += weights["interests"]
    
    # compare similar values if it exists in each person
    if "values" in person1 and "values" in person2:
        values_list1 = person1["values"]
        values_list2 = person2["values"]
        
        for value1 in values_list1:
            for value2 in values_list2:
                stem1 = ss.stem(value1.lower())
                stem2 = ss.stem(value2.lower())
            
            if stem1 == stem2:
                score += weights["values"]
                
    # compare age similarity if it exists for each person
    if "age" in person1 and "age" in person2:
        age1 = float(person1["age"])
        age2 = float(person2["age"])
        
        score -= abs(age1 - age2) * weights["age"]
    
    # score proximity of the paired couple if coordinates exists for each person
    if "coordinates" in person1 and "coordinates" in person2:
        coord_inputs = {
            "lat1": person1["coordinates"]["lat"],
            "lon1": person1["coordinates"]["long"],
            "lat2": person2["coordinates"]["lat"],
            "lon2": person2["coordinates"]["long"],
            "type": "miles"
            }
        distance = client.algo("geo/GeoDistance").pipe(coord_inputs).result
        #print "distance: {}".format(distance)
        #print "weights: {}".format(weights)
        score -= distance * weights["coordinates"]
    
    return score
    
def validateInput(input):
    # Validate the initial input fields
    if "group1" not in input and "group2" not in input:
        raise AlgorithmError("Please provide both the male and female groups")
    elif "group2" not in input:
        raise AlgorithmError("Please provide the female group.")
    elif "group1" not in input:
        raise AlgorithmError("Please provide the male group.")
    
    # The only required field for a user object is "name" and "interests"
    for gender in ["group1", "group2"]:
        if not isinstance(input[gender], list):
            raise AlgorithmError("Please provide a list of people for each group.")
        
        if len(input[gender]) == 0:
            raise AlgorithmError("Groups cannot be empty.")
        
        for person in input[gender]:
            if "name" not in person or "interests" not in person:
                raise AlgorithmError("Please provide the name and interests for all people.")
            
            if not isinstance(person["interests"], list):
                raise AlgorithmError("Please provide a list of interests for each person.")
                
            # Check validity for the longitude and latitude if the coordinates field exists
            if "coordinates" in person:
                if not isinstance(person["coordinates"], dict):
                    raise AlgorithmError("Please provide valid coordinates")
                if "lat" not in person["coordinates"] or "long" not in person["coordinates"]:
                    raise AlgorithmError("Please provide valid coordinates")
                if not isinstance(person["coordinates"]["lat"], float) or not isinstance(person["coordinates"]["long"], float):
                    raise AlgorithmError("coordinate values can only be in float.")
    
    # unequal groups are now supported
    # if len(input["group1"]) != len(input["group2"]):
    #     raise AlgorithmError("The size of both groups should be same.")
print(apply(example_input))