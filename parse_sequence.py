def parse_sequence(response):
    parts = [subsubpart for part in response.split(',')
             for subpart in part.strip().split('-')
             for subsubpart in subpart.split(' ') if len(subsubpart) == 1]
    return parts
