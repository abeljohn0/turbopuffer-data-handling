import { hybridSearch, keywordSearch, tpuf, uploadUserFiles, vectorSearch } from "./turbopuffer";
import { embedVoyageContextInput, InputType } from "./voyage";

const query =
  "What are the requirements and procedures for filing a Form 8832 Entity Classification Election, and what extensions of time are available under Section 301.9100-3?";

const testHybridSearch = async () => {
  const embedding = await embedVoyageContextInput(
    {
      query: query,
    },
    InputType.QUERY
  );
  const rows = await hybridSearch("content-prod", 10, query, embedding[0]);
  console.log(rows[0]);
  console.log("testHybridSearch done ✅");
};

const testVectorSearch = async () => {
  const rows1 = await keywordSearch("content-prod", 10, query);
  console.log(rows1[0]);

  const embedding = await embedVoyageContextInput(
    {
      query: query,
    },
    InputType.QUERY
  );
  const rows2 = await vectorSearch("content-prod", 10, embedding[0]);
  console.log(rows2[0]);
  console.log("testVectorSearch done ✅");
};

const testKeywordSearch = async () => {
  const rows = await keywordSearch("content-prod", 10, query);
  console.log(rows[0]);
  console.log("testKeywordSearch done ✅");
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const testUploadMultipleUserFiles = async () => {
  const output = await uploadUserFiles("user-test-ns", [
    {
      displayName: "test2",
      fileId: "456",
      markdown:
        "# §4978A Repealed. Pub. L. 101-239, title VII, §7304(a)(2)(C)(i), Dec. 19, 1989, 103 Stat. 2353 Section, added Pub. L. 100-203, title X, §10413(a), Dec. 22, 1987, 101 Stat. 1330-436; amended Pub. L. 100-647, title VI, §6060(a), Nov. 10, 1988, 102 Stat. 3699, related to tax on certain dispositions of employer securities to which [section 2057](!cite_dd11614a-d569-4dc0-9e18-06951da6c1a1) applied.",
    },
    {
      displayName: "test3",
      fileId: "789",
      markdown:
        "See Goldin v . Commissioner, T .C . Memo. 2004-129; Brown v. Commissioner, T .C . Memo. 2002-187. We lack jurisdiction to grant petitioner relief under section 6013(e) for any of the years 1989 through and 1992. Accordingly, An order will be issued striking the portions of the petition requesting relief under sections 66(c) and 6013(e) and dismissing the related portions of this case .",
    },
  ]);
  console.log(output);
  const ns = tpuf.namespace("user-test-ns");
  await sleep(5000);
  const rows = await ns.query({
    rank_by: ["id", "asc"],
    top_k: 10,
    include_attributes: true,
  });
  console.log(rows.rows);
  console.log("testUploadUserFiles done ✅");
};

const testUploadSingleUserFile = async () => {
  const output = await uploadUserFiles("user-test-ns", [
    {
      displayName: "test2",
      fileId: "456",
      markdown:
        "# §4978A Repealed. Pub. L. 101-239, title VII, §7304(a)(2)(C)(i), Dec. 19, 1989, 103 Stat. 2353 Section, added Pub. L. 100-203, title X, §10413(a), Dec. 22, 1987, 101 Stat. 1330-436; amended Pub. L. 100-647, title VI, §6060(a), Nov. 10, 1988, 102 Stat. 3699, related to tax on certain dispositions of employer securities to which [section 2057](!cite_dd11614a-d569-4dc0-9e18-06951da6c1a1) applied.",
    },
  ]);
  console.log(output[0]);
  const ns = tpuf.namespace("user-test-ns");
  await sleep(3000);
  const rows = await ns.query({
    rank_by: ["id", "asc"],
    top_k: 10,
    include_attributes: true,
  });
  console.log(rows.rows);
  console.log("testUploadUserFiles done ✅");
};

// testUploadMultipleUserFiles();
// testUploadSingleUserFile();
// testHybridSearch();
// testVectorSearch();
// testKeywordSearch();