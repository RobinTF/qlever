include(GoogleTest)

# Needed for creating the `testUil`-library.
add_subdirectory(util EXCLUDE_FROM_ALL)

# Link binary ${basename} against `gmock_main`, the threading library, the
# general test utilities and all libraries that are specified as additional
# arguments.
function(linkTest basename)
    qlever_target_link_libraries(${basename} ${ARGN} GTest::gtest GTest::gmock_main ${CMAKE_THREAD_LIBS_INIT})
endfunction()

# Add the executable ${basename} that is compiled from the source file
# "${basename}".cpp
function(addTest basename)
    add_executable(${basename} "${basename}.cpp")
endfunction()

# Usage: `linkAndDiscoverTest(basename, [additionalLibraries...]`
# Link the executable `basename` against `gmock_main`,threading library,
# and all `additionLibraries` which are passed as arguments.
# Then run `gtest_discover_tests` to add the tests cases from the executable.
# Typically you should use `addAndLinkTest` (below) but this function can be used,
# if a test binary requires multiple sources
function(linkAndDiscoverTest basename)
    linkTest(${basename} ${ARGN})
    gtest_discover_tests(${basename} ${basename} DISCOVERY_TIMEOUT 600)
endfunction()

# Usage: `linkAndDiscoverTestSerial(basename, [additionalLibraries...]`
# Similar to `linkAndDiscoverTestSerial` but also requires that the test is run serially
# (without any of the other test cases running in parallel). This can be
# required e.g. if several tests cases write to the same file.
function(linkAndDiscoverTestSerial basename)
    linkTest(${basename} ${ARGN})
    gtest_discover_tests(${basename} ${basename} DISCOVERY_TIMEOUT 600 PROPERTIES RUN_SERIAL
            TRUE)
endfunction()

if (SINGLE_TEST_BINARY)
    message(STATUS "All tests are linked into a single executable `QLeverAllUnitTestsMain`")
    add_executable(QLeverAllUnitTestsMain)
    qlever_target_link_libraries(QLeverAllUnitTestsMain gtest gmock_main testUtil ${CMAKE_THREAD_LIBS_INIT})
    # The following line (as opposed to using `gtest_discover_tests`) has the effect that `ctest` will treat all the
    # unit tests as a single test case and will simply run the executable (using the main function from `gtest_main`.
    # This decreases the runtime overhead per unit tests in particular for sanitizer builds.
    add_test(NAME QLeverAllUnitTests COMMAND $<TARGET_FILE:QLeverAllUnitTestsMain>)
else ()
    message(STATUS "The tests are split over multiple binaries")

endif ()

# The implementation of `addLinkAndDiscoverTest` and `addLinkandDiscoverTestNoLibs` below.
function(addLinkAndDiscoverTestImpl basename)
    if (SINGLE_TEST_BINARY)
        target_sources(QLeverAllUnitTestsMain PUBLIC ${basename}.cpp)
        qlever_target_link_libraries(QLeverAllUnitTestsMain ${ARGN} )
    else ()
        addTest(${basename})
        linkAndDiscoverTest(${basename} ${ARGN})
    endif ()

endfunction()

# Usage: `addAndLinkTest[NoLibs](basename, [additionalLibraries...]`
# Add a GTest/GMock test case that is called `basename` and compiled from a file called
# `basename.cpp`. All tests are linked against `gmock_main` and the threading library.
# additional libraries against which the test case has to be linked can be specified as
# additional arguments after the `basename`

# This function links the test against `testUtil` (basically all of QLever).
function(addLinkAndDiscoverTest basename)
    addLinkAndDiscoverTestImpl(${basename} ${ARGN} testUtil)
endfunction()

# This function links only against Gtest + the explicitly specified libraries.
# It can be used for tests of standalone utils that don't require the rest of QLever.
function(addLinkAndDiscoverTestNoLibs basename)
    addLinkAndDiscoverTestImpl(${basename} ${ARGN})
endfunction()


# Add a GTest/GMock test case that is called `basename` and compiled from a file called
# `basename.cpp`. All tests are linked against `gmock_main` and the threading library.
# In contrast to `addLinkAndDiscoverTest` this doesn't let ctest run all subtests individually,
# but all at once to reduce overhead in the CI pipeline.
function(addLinkAndRunAsSingleTest basename)
    if (SINGLE_TEST_BINARY)
        target_sources(QLeverAllUnitTestsMain PUBLIC ${basename}.cpp)
        qlever_target_link_libraries(QLeverAllUnitTestsMain ${ARGN})
    else ()
        addTest(${basename})
        linkTest(${basename} testUtil ${ARGN})
        add_test(NAME ${basename} COMMAND ${basename})
    endif ()

endfunction()

# Usage: `addAndLinkTestSerial(basename, [additionalLibraries...]`
# Similar to `addAndLinkTest` but also requires that the test is run serially
# (without any of the other test cases running in parallel). This can be
# required e.g. if several tests cases write to the same file.
function(addLinkAndDiscoverTestSerial basename)
    if (SINGLE_TEST_BINARY)
        target_sources(QLeverAllUnitTestsMain PUBLIC ${basename}.cpp)
        qlever_target_link_libraries(QLeverAllUnitTestsMain ${ARGN})
    else ()
        addTest(${basename})
        linkAndDiscoverTestSerial(${basename} testUtil ${ARGN})
    endif ()
endfunction()

# Only compile and link the test, but do not run it.
# Usage: Same as for the two functions above.
function(addAndLinkTest basename)
    addTest(${basename})
    linkTest(${basename} ${ARGN})
endfunction()

add_subdirectory(engine)
add_subdirectory(parser)
add_subdirectory(index)
add_subdirectory(backports)
add_subdirectory(rdfTypes)
add_subdirectory(joinAlgorithms)

addLinkAndDiscoverTest(ValueIdComparatorsTest util)

addLinkAndDiscoverTest(SparqlParserTest parser engine sparqlExpressions)

addLinkAndDiscoverTest(StringUtilsTest util)

addLinkAndDiscoverTest(CryptographicHashUtilsTest util)

addLinkAndDiscoverTest(CacheTest)

addLinkAndDiscoverTestNoLibs(ConcurrentCacheTest)

# This test also seems to use the same filenames and should be fixed.
addLinkAndDiscoverTestSerial(FileTest)

addLinkAndDiscoverTest(Simple8bTest)

addLinkAndDiscoverTest(WordsAndDocsFileParserTest parser)

addLinkAndDiscoverTest(IndexMetaDataTest index)

# We currently always use static file names for all indices, which
# makes it impossible to run the test cases for the Index class in parallel.
# TODO<qup42, joka921> fix this
addLinkAndDiscoverTestSerial(IndexTest index)

addLinkAndDiscoverTest(LocatedTriplesTest index)

addLinkAndDiscoverTestSerial(IdTripleTest index)

addLinkAndDiscoverTestSerial(DeltaTriplesTest index)

addLinkAndDiscoverTest(DeltaTriplesCountTest index)

addLinkAndDiscoverTest(EngineTest engine)

addLinkAndDiscoverTest(JoinTest engine)

addLinkAndDiscoverTest(TextLimitOperationTest engine)

addLinkAndDiscoverTestSerial(QueryPlannerTest engine)

addLinkAndDiscoverTestNoLibs(HashMapTest)

addLinkAndDiscoverTest(HashSetTest)

addLinkAndDiscoverTestSerial(GroupByTest engine)

addLinkAndDiscoverTest(VocabularyGeneratorTest index)

addLinkAndDiscoverTest(HasPredicateScanTest engine)

addLinkAndDiscoverTest(MmapVectorTest)

# BufferedVectorTest also uses conflicting filenames.
addLinkAndDiscoverTestSerial(BufferedVectorTest)

addLinkAndDiscoverTest(UnionTest engine)

if (SINGLE_TEST_BINARY)
    target_sources(QLeverAllUnitTestsMain PUBLIC TokenTest.cpp TokenTestCtreHelper.cpp)
    qlever_target_link_libraries(QLeverAllUnitTestsMain parser re2 util)
else ()
    add_executable(TokenTest TokenTest.cpp TokenTestCtreHelper.cpp)
    linkAndDiscoverTest(TokenTest parser re2 util)
endif ()

addLinkAndDiscoverTestSerial(RdfParserTest parser re2)

addLinkAndDiscoverTest(MultiColumnJoinTest engine)

addLinkAndDiscoverTest(IdTableTest util)

addLinkAndDiscoverTest(TransitivePathTest engine)

addLinkAndDiscoverTest(PathSearchTest engine)

addLinkAndDiscoverTest(BatchedPipelineTest)

addLinkAndDiscoverTest(TupleHelpersTest)

addLinkAndDiscoverTestNoLibs(StringSortComparatorTest)

addLinkAndDiscoverTest(PriorityQueueTest)

addLinkAndDiscoverTest(SynchronizedTest)

addLinkAndDiscoverTest(AllocatorWithLimitTest)

addLinkAndDiscoverTest(MinusTest engine)

# this test runs for quite some time and might have spurious failures!
# Therefore it is compiled, but not run. If you want to run it,
# change the following two lines.
addAndLinkTest(SortPerformanceEstimatorTest SortPerformanceEstimator)
#addLinkAndDiscoverTest(SortPerformanceEstimatorTest SortPerformanceEstimator)

# The SerializerTest uses temporary files. The tests fail when multiple test
# cases are run in parallel. This should be fixed by using distinct filenames
# for each test case.
# TODO<qup42, joka921> fix this
addLinkAndDiscoverTestSerial(SerializerTest)

addLinkAndDiscoverTestNoLibs(ParametersTest)

addLinkAndDiscoverTest(ZstdCompressionTest zstd ${cmake_thread_libs_init})

addLinkAndDiscoverTest(TaskQueueTest)

addLinkAndDiscoverTest(SetOfIntervalsTest sparqlExpressions)

addLinkAndDiscoverTestNoLibs(TypeTraitsTest)

addLinkAndDiscoverTestSerial(SparqlExpressionTest sparqlExpressions index engine)

addLinkAndDiscoverTest(StreamableBodyTest http)

addLinkAndDiscoverTest(StreamableGeneratorTest)

addLinkAndDiscoverTest(AcceptHeaderTest mediaTypes httpParser)

addLinkAndDiscoverTestNoLibs(CompactStringVectorTest)

addLinkAndDiscoverTest(SparqlDataTypesTest engine)

addLinkAndDiscoverTest(ContentEncodingHelperTest http)


addLinkAndDiscoverTest(PrefixCompressorTest)

addLinkAndDiscoverTest(VocabularyTest index)

addLinkAndDiscoverTestNoLibs(IteratorTest)

addLinkAndDiscoverTestNoLibs(ViewsTest)

addLinkAndDiscoverTest(ForwardTest)

addLinkAndDiscoverTest(CompressorStreamTest engine)

addLinkAndDiscoverTestNoLibs(AsyncStreamTest)

addLinkAndDiscoverTest(BitUtilsTest)

addLinkAndDiscoverTest(NBitIntegerTest)

addLinkAndDiscoverTest(GeoPointTest)

addLinkAndDiscoverTest(GeoSparqlHelpersTest util)

addLinkAndDiscoverTest(HttpUtilsTest util http)

addLinkAndDiscoverTestSerial(DateYearDurationTest util parser)

addLinkAndDiscoverTest(DurationTest util)

addLinkAndDiscoverTest(TripleComponentTest parser)

addLinkAndDiscoverTest(ValueIdTest util)

addLinkAndDiscoverTestNoLibs(LambdaHelpersTest)

addLinkAndDiscoverTest(ParseExceptionTest parser engine)

addLinkAndDiscoverTest(TransparentFunctorsTest)

addLinkAndDiscoverTest(SelectClauseTest parser engine)

addLinkAndDiscoverTestSerial(RelationalExpressionTest parser sparqlExpressions index engine)

addLinkAndDiscoverTest(CheckUsePatternTrickTest parser engine)

addLinkAndDiscoverTestSerial(RegexExpressionTest parser sparqlExpressions index engine parser)

addLinkAndDiscoverTestSerial(LocalVocabTest engine)

addLinkAndDiscoverTestSerial(ValuesTest engine)

addLinkAndDiscoverTestSerial(ServiceTest engine)

addLinkAndDiscoverTestSerial(LoadTest engine)

addLinkAndDiscoverTestSerial(HttpTest Boost::iostreams http)

addLinkAndDiscoverTestNoLibs(CallFixedSizeTest)

addLinkAndDiscoverTest(ConstexprUtilsTest)

addLinkAndDiscoverTestNoLibs(ResetWhenMovedTest)

addLinkAndDiscoverTest(TimerTest)

addLinkAndDiscoverTest(AlgorithmTest)

addLinkAndDiscoverTestSerial(CompressedRelationsTest index)

addLinkAndDiscoverTestSerial(PrefilterExpressionIndexTest engine)

addLinkAndDiscoverTestSerial(GetPrefilterExpressionFromSparqlExpressionTest sparqlExpressions index)

addLinkAndDiscoverTest(ExceptionTest)

addLinkAndDiscoverTestSerial(RandomExpressionTest index)

addLinkAndDiscoverTestSerial(NowDatetimeExpressionTest index)

addLinkAndDiscoverTestSerial(LanguageExpressionsTest index)

addLinkAndDiscoverTestSerial(SortTest engine)

addLinkAndDiscoverTestSerial(OrderByTest engine)

addLinkAndDiscoverTestSerial(ValuesForTestingTest index)

addLinkAndDiscoverTestSerial(ExportQueryExecutionTreesTest index engine parser)

addLinkAndDiscoverTestSerial(AggregateExpressionTest parser sparqlExpressions index engine)

addLinkAndDiscoverTest(OnDestructionDontThrowDuringStackUnwindingTest)

addLinkAndDiscoverTest(ExceptionHandlingTest)

addLinkAndDiscoverTest(SparqlExpressionTypesTest sparqlExpressions util)

addLinkAndDiscoverTest(LimitOffsetClauseTest)

addLinkAndDiscoverTest(OperationTest engine)

addLinkAndDiscoverTest(RuntimeInformationTest engine index)

addLinkAndDiscoverTest(VariableToColumnMapTest parser)

addLinkAndDiscoverTest(CopyableUniquePtrTest)

addLinkAndDiscoverTest(JsonCustomConverterForThirdPartyTest)

addLinkAndDiscoverTest(ConfigManagerTest configManager)

addLinkAndDiscoverTest(ConfigOptionTest configManager)

addLinkAndDiscoverTest(ValidatorTest configManager)

addLinkAndDiscoverTest(ConfigOptionProxyTest configManager)

addLinkAndDiscoverTest(ConfigUtilTest configManager)

addLinkAndDiscoverTest(RandomTest)

addLinkAndDiscoverTest(BenchmarkMeasurementContainerTest benchmark testUtil)

addLinkAndDiscoverTest(ResultTableColumnOperationsTest benchmark testUtil)

addLinkAndDiscoverTest(FindUndefRangesTest engine)

addLinkAndDiscoverTest(AddCombinedRowToTableTest engine util)

addLinkAndDiscoverTest(CtreHelpersTest)

addLinkAndDiscoverTest(ComparisonWithNanTest)

addLinkAndDiscoverTestNoLibs(ThreadSafeQueueTest)

addLinkAndDiscoverTest(IdTableHelpersTest)

addLinkAndDiscoverTest(GeneratorTest)

addLinkAndDiscoverTest(MemorySizeTest memorySize)

addLinkAndDiscoverTest(JsonUtilTest)

addLinkAndDiscoverTest(JoinAlgorithmsTest)

addLinkAndDiscoverTest(AsioHelpersTest)

addLinkAndDiscoverTest(UniqueCleanupTest)

addLinkAndDiscoverTest(WebSocketSessionTest http)

addLinkAndDiscoverTest(QueryIdTest)

addLinkAndDiscoverTest(QueryHubTest http)

addLinkAndDiscoverTest(QueryToSocketDistributorTest http)

addLinkAndDiscoverTest(UpdateFetcherTest http)

addLinkAndDiscoverTest(MessageSenderTest http)

addLinkAndDiscoverTestNoLibs(CancellationHandleTest util)

addLinkAndDiscoverTest(ProgressBarTest util)

addLinkAndDiscoverTest(CachingMemoryResourceTest)

addLinkAndDiscoverTestNoLibs(ParallelMultiwayMergeTest)

addLinkAndDiscoverTest(ParseableDurationTest)

addLinkAndDiscoverTest(ConstantsTest)

addLinkAndDiscoverTest(JThreadTest)

addLinkAndDiscoverTest(ChunkedForLoopTest)

addLinkAndDiscoverTest(FsstCompressorTest fsst)

addLinkAndDiscoverTest(CopyableSynchronizationTest)

addLinkAndDiscoverTest(LazyJsonParserTest)

addLinkAndDiscoverTest(GeneratorsTest)

addLinkAndDiscoverTest(FilterTest engine)

addLinkAndDiscoverTest(ResultTest engine)

addLinkAndDiscoverTest(BlankNodeManagerTest)

addLinkAndDiscoverTest(SparqlExpressionGeneratorsTest engine)

addLinkAndDiscoverTest(UrlParserTest)

addLinkAndDiscoverTest(ServerTest engine)

addLinkAndDiscoverTest(ExecuteUpdateTest engine)

addLinkAndDiscoverTest(ValueGetterTest engine)

addLinkAndDiscoverTest(GraphStoreProtocolTest engine)

addLinkAndDiscoverTest(SparqlProtocolTest engine)

addLinkAndDiscoverTest(ParsedRequestBuilderTest engine)

addLinkAndDiscoverTest(LruCacheTest)

addLinkAndDiscoverTestNoLibs(InputRangeUtilsTest)

addLinkAndDiscoverTest(TripleSerializerTest)

addLinkAndDiscoverTest(GeometryInfoTest util)

addLinkAndDiscoverTestSerial(QueryRewriteUtilTest engine)

addLinkAndDiscoverTestSerial(HttpErrorTest engine)
