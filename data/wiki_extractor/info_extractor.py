import wiki_api
import time

def main():


    search_inputs = ["console game",    "game company logos",    "video games",    "video game console",    "arcade game",    "computer game",    "mobile game",    "Nintendo game",    "PlayStation game",    "Xbox game",    "PC game",    "retro game",    "video game characters",    "video game controllers",    "video game cover art",    "video game development",    "video game fan art",    "video game merchandise",    "video game music",    "video game screenshots",    "video game sprites",    "video game trailers",    "virtual reality game",    "online game",    "game ratings",    "multiplayer games",    "single player games",    "action games",    "adventure games",    "role-playing games",    "sports games",    "strategy games",    "simulation games",    "puzzle games",    "shooter games",    "platformer games",    "fighting games",    "racing games",    "open world games",    "sandbox games",    "stealth games",    "survival games",    "horror games",    "educational games",    "game development tools",    "game engines",    "game modding",    "game design",    "gaming culture",    "esports",    "gaming events",    "gaming news"]



    # open file to write descriptions
    with open("../descriptions.txt", "w") as f:

        # loop through search inputs
        image_index = 0
        i = 0
        while i < len(search_inputs):
            search_input = search_inputs[i]
            i+=1
            print(f"Searching for '{search_input}'")

            # search for input and get results
            results = wiki_api.search(search_input, number_of_results=100, project="commons", language="")
            
            
            pages = results.get('pages', None)
            
            # if there are no pages, wait 30 minutes and try again
            if pages is None:
                print(results)
                print("ERROR: No pages found")
                time.sleep(1800)
                i -= 1
                continue

            # loop through results
            j = 0
            while j < len(pages):
                
                result = pages[j]
                j+=1
                # get resource info
                key: str = result.get('key', None)

                # if there is no key, there is an error with the request, so wait 30 minutes and try again
                if key is None:
                    time.sleep(1800)
                    i -= 1
                    continue

                # get image extension and continue if it is not jpg, png or svg
                extension = key[-3:]
                if extension not in ["jpg", "png"]:
                    continue

                info = wiki_api.get_resource_info(key, project="commons")

                # get description url
                description_url = info.get('file_description_url', None)

                # if there is no description url, continue
                if description_url is None:
                    continue

                # add protocol to url and get description
                description_url = f"https:{description_url}"
                description = wiki_api.get_resource_description(description_url)

                # if there is no description, continue
                if description is None:
                    continue
                
                # write description to file
                f.write(f"{description}")
                f.write('\n-\n')
 

                # download image
                wiki_api.download_resource(f"{image_index:05}_{key[5:]}", project="commons", destination_path="../images", resource_info=info)

                # increment image index
                image_index += 1


            f.flush()



if __name__ == "__main__":
    main()