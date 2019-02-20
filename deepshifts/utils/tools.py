def as_xyz(species, coords, title='title'):
    lines = [str(len(species)), title]
    for s,(x,y,z) in zip(species,coords):
        lines.append('%-6s %12.7f %12.7f %12.7f'%(s,x,y,z))
    return '\n'.join(lines)