"""The probability map: where the missing person may be (POA), how well each cell has been
seen (POD), and which sector to search next (PSR).

Pure logic without I/O: the search service feeds it and writes the result to the ground
station. Distance rings per subject category plus terrain weights give the prior; each
pass without a find lowers the searched cells with Bayes' rule.
"""
