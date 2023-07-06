
def remove_comments(code):
    def non_comment(line):
        if len(line) == 0:
            return False
        return not (line[0][0] == "*" or line[0] == "/**")

    return "\n".join(
        list(
            filter(
                lambda s: non_comment(
                    list(filter(lambda ss: len(ss) > 0, s.split(" ")))
                ),
                code.splitlines(),
            )
        )
    )